import code

import torch
import torch.nn as nn
import numpy as np

from components.distributions import Delta
import components.pytorch_util as ptu
from models.q_networks import Q_Network
from models.policy_networks import TanhGaussianPolicy
from trainers.base_trainer import BaseTrainer

class SAC(BaseTrainer):

    def init_model(self):
        self.print("init SAC model")
        self.qf1 = Q_Network(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.qf2 = Q_Network(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.target_qf1 = Q_Network(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.target_qf2 = Q_Network(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.policy = TanhGaussianPolicy(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())
        self.q_learner = self.qf1
        # optimizers
        self.qf1_optimizer = torch.optim.Adam(
            self.qf1.parameters(),
            lr=self.args.lr
        )
        self.qf2_optimizer = torch.optim.Adam(
            self.qf2.parameters(),
            lr=self.args.lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args.lr
        )
        # temperature function (optional)
        if self.args.use_temperature_func:
            self.log_alpha = torch.zeros(1, device=self.args.device, requires_grad=True)
            if self.args.target_entropy is None:
                self.target_entropy = -np.prod(self.expl_env.action_space.shape).item()
            else:
                self.target_entropy = self.args.target_entropy
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=self.args.lr
            )

    def expl_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dist = self.policy(obs)
            a = dist.sample()
            return a.cpu()

    def eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ get evaluation action """
        with torch.no_grad():
            deterministic_dist = Delta(self.policy(obs).mle_estimate())
            a = deterministic_dist.sample()
            return a.cpu()

    def before_learn_on_batch(self):
        # PER (prioritized experience replay) anneal
        if self.args.prioritized:
            self.replay_buffer.anneal_priority_weight()

    def learn_on_batch(self, *args):
        """
            s: (b, s_dim)
            a: (b, a_dim)
            r: (b,)
            s_: (s_dim)
            d: (b, 1)
            w: (b, )
        """
        if self.args.prioritized:
            treeidx, s, a, r, s_, d, w = args
            w = torch.unsqueeze(w, dim=-1)
        else:
            s, a, r, s_, d = args
        r = torch.unsqueeze(r, dim=-1)
        # policy and alpha loss (optional)
        dist = self.policy(s)
        new_state_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(dim=-1)   # (b, 1)
        if self.args.use_temperature_func:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.args.alpha

        q_new_actions = torch.min(
            self.qf1(s, new_state_actions),
            self.qf2(s, new_state_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        # Q loss
        q1_pred = self.qf1(s, a)
        q2_pred = self.qf2(s, a)
        with torch.no_grad():
            next_dist = self.policy(s_)
            new_next_state_actions, next_log_pi = next_dist.rsample_and_logprob()
            next_log_pi = next_log_pi.unsqueeze(dim=-1)   # (b, 1)
            target_q_values = torch.min(
                self.target_qf1(s_, new_next_state_actions),
                self.target_qf2(s_, new_next_state_actions),
            ) - alpha * next_log_pi   # (b, 1)
            q_target = r + (1. - d) * self.args.discount * target_q_values   # (b, 1)
        qf1_loss = (q1_pred - q_target) ** 2
        qf2_loss = (q2_pred - q_target) ** 2

        # backward
        if self.args.use_temperature_func:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.args.prioritized:
            # PER Importance Sampling (IS)
            IS_mseloss1 = torch.mean(qf1_loss * w)
            IS_mseloss2 = torch.mean(qf2_loss * w)

            self.qf1_optimizer.zero_grad()
            IS_mseloss1.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            IS_mseloss2.backward()
            self.qf2_optimizer.step()

            # PRE update
            abs_loss = ((torch.sqrt(qf1_loss) + torch.sqrt(qf2_loss)) / 2).detach().view(-1).cpu().numpy()
            self.replay_buffer.update_priorities(treeidx, abs_loss)
        else:
            self.qf1_optimizer.zero_grad()
            qf1_loss.mean().backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.mean().backward()
            self.qf2_optimizer.step()

        # logging
        with torch.no_grad():
            if self.args.prioritized:
                self.exp_path['batch_stats[t,num_done,pred_mean,q_loss,target_mean]'].csv_writerow(
                    [self.t, d.sum().item(), new_state_actions.mean().item(), abs_loss.mean().item(), q_target.mean().item()])
            else:
                self.exp_path['batch_stats[t,num_done,pred_mean,q_loss,target_mean]'].csv_writerow(
                    [self.t, d.sum().item(), new_state_actions.mean().item(), qf1_loss.mean().item(), q_target.mean().item()])

    def after_learn_on_batch(self):
        # update target network
        if self.t % self.args.target_update_freq == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.args.tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.args.tau
            )

    def state_value_pred(self, s):
        """
        log the state value of initial state
            s: (s_dim, )
        """
        with torch.no_grad():
            s = s.unsqueeze(dim=0)  # (1, s_dim)
            dist = self.policy(s)
            sampled_actions, log_pi = dist.rsample_and_logprob()
            log_pi = log_pi.unsqueeze(dim=-1)   # (1, 1)
            if self.args.use_temperature_func:
                alpha = torch.exp(self.log_alpha)
            else:
                alpha = self.args.alpha
            target_q_values = torch.min(
                self.target_qf1(s, sampled_actions),
                self.target_qf2(s, sampled_actions),
            ) - alpha * log_pi   # (1, 1)
            return torch.unsqueeze(target_q_values, dim=0).item()
