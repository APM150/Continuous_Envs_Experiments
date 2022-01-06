import code

import torch
import numpy as np

import components.memory
import components.prioritized_memory
from trainers.base_trainer import BaseTrainer
from models.q_networks import Ensemble_Q_Network
from models.policy_networks import TanhGaussianPolicy
from components.distributions import Delta
import components.pytorch_util as ptu


class Mean_DSPG(BaseTrainer):
    """ Deep Stochastic Policy Gradient """

    def init_model(self):
        self.print("init Mean DSPG model")
        self.q_learner = Ensemble_Q_Network(
            args=self.args,
            obs_dim=self.expl_env.observation_space.shape[0],
            action_dim=self.expl_env.action_space.shape[0],
        ).to(self.args.device)
        self.q_target = Ensemble_Q_Network(
            args=self.args,
            obs_dim=self.expl_env.observation_space.shape[0],
            action_dim=self.expl_env.action_space.shape[0],
        ).to(self.args.device)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.policy = TanhGaussianPolicy(
            self.args,
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        # optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_learner.parameters(),
            lr=self.args.q_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args.p_lr
        )
        self.replay_buffer = components.memory.EnsembleReplayMemory(self.args)

    def expl_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ get exploration action """
        with torch.no_grad():
            dist = self.policy(obs)
            action = dist.sample()
            return action.cpu()

    def eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        """ get evaluation action """
        with torch.no_grad():
            deterministic_dist = Delta(self.policy(obs).mle_estimate())
            action = deterministic_dist.sample()
            return action.cpu()

    def before_learn_on_batch(self):
        # PER (prioritized experience replay) anneal
        if self.args.prioritized:
            self.replay_buffer.anneal_priority_weight()

    def target_estimate(self, s_):
        b, k, s_dim = s_.shape[0], s_.shape[1], s_.shape[2]
        s_new = s_.reshape(b*k, s_dim)  # (bk, s)
        with torch.no_grad():
            next_dist = self.policy(s_new)
            next_action, _ = next_dist.rsample_and_logprob()  # (bk, a)
            target_pred = self.q_target(s_new, next_action)  # (bk, k, 1)
            next_state_value = target_pred.view(b, k, k, 1).mean(dim=2)  # (b, k, 1)
            return next_state_value

    def learn_on_batch(self, *args):
        """
        learn on batch, all arguments are torch.Tensor
            s: (b, k, s_dim)
            a: (b, k, a_dim)
            r: (b, k)
            s_:(b, k, s_dim)
            d: (n, k, 1)
        """
        if self.args.prioritized:
            treeidx, s, a, r, s_, d, w = args
            w = torch.unsqueeze(w, dim=-1)
        else:
            s, a, r, s_, d = args
        r = torch.unsqueeze(r, dim=-1)
        b, k, s_dim = s.shape[0], s.shape[1], s.shape[2]
        s_new = s.reshape(b*k, s_dim)
        # policy loss
        dist = self.policy(s_new)
        policy_action, _ = dist.rsample_and_logprob()  # (bk, a)
        q_output = self.q_learner(s_new, policy_action)  # (bk, k, 1)
        q_output = q_output.view(b, k, k, 1).mean(dim=2)  # (b, k, 1)
        policy_loss = -q_output.mean()

        # Q loss
        next_state_value = self.target_estimate(s_)  # (b, k, 1)
        q_target = r + (1. - d) * self.args.discount * next_state_value
        q_pred = self.q_learner(s, a)  # (b, k, 1)
        qf_loss = (q_pred - q_target) ** 2

        # backward
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.q_optimizer.zero_grad()
        qf_loss.mean().backward()
        self.q_optimizer.step()

        # log stats
        with torch.no_grad():
            self.exp_path['batch_stats[t,q_loss,policy_loss]'].csv_writerow(
                [self.t, qf_loss.mean().item(), policy_loss.item()]
            )
            self.exp_path['batch_stats[t,batch_policy_std]'].csv_writerow(
                [self.t, dist.normal_std.mean().item()]
            )

    def after_learn_on_batch(self):
        # update target network
        if self.t % self.args.target_update_freq == 0:
            ptu.soft_update_from_to(
                self.q_learner, self.q_target, self.args.tau
            )

    def state_value_pred(self, s):
        """
        log the state value of initial state
            s: (s_dim, )
        """
        with torch.no_grad():
            s = s.unsqueeze(dim=0)  # (1, s_dim)
            dist = self.policy(s)
            action, _ = dist.rsample_and_logprob()  # (1, a)
            target_q_values = self.q_target(s, action).mean(dim=1) # (1, 1)
        return torch.unsqueeze(target_q_values, dim=0).item()
