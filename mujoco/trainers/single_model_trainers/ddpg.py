import code

import torch
import torch.nn as nn
import numpy as np

import components.pytorch_util as ptu
from models.q_networks import Q_Network
from models.policy_networks import TanhMlpPolicy
from trainers.base_trainer import BaseTrainer
from components.utils import OUProcess

class DDPG(BaseTrainer):

    def init_model(self):
        self.print("init DDPG model")
        self.q_learner = Q_Network(
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.q_target = Q_Network(
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.q_target.load_state_dict(self.q_learner.state_dict())
        self.policy = TanhMlpPolicy(
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.policy_target = TanhMlpPolicy(
            self.expl_env.observation_space.shape[0],
            self.expl_env.action_space.shape[0]
        ).to(self.args.device)
        self.policy_target.load_state_dict(self.policy.state_dict())
        # optimizers
        self.q_optimizer = torch.optim.Adam(
            self.q_learner.parameters(),
            lr=self.args.q_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.args.p_lr
        )
        self.ou_process = OUProcess(self.expl_env.action_space.shape[0])

    def expl_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = self.policy(obs)
            a = self.ou_process.get_action_from_raw_action(a.cpu(), self.t)
            return a

    def eval_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = self.policy(obs)
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
        # policy loss
        policy_action = self.policy(s)  # (b, a)
        q_output = self.q_learner(s, policy_action)  # (b, 1)
        policy_loss = - q_output.mean()

        # Q loss
        with torch.no_grad():
            next_actions = self.policy_target(s_)  # (b, a)
            target_q_values = self.q_target(s_, next_actions) # (b, 1)
            q_target = r + (1. - d) * self.args.discount * target_q_values
        q_pred = self.q_learner(s, a)  # (b, 1)
        qf_loss = (q_pred - q_target) ** 2

        # backward
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.q_optimizer.zero_grad()
        qf_loss.mean().backward()
        self.q_optimizer.step()

        # logging
        with torch.no_grad():
            self.exp_path['batch_stats[t,q_loss,policy_loss]'].csv_writerow(
                [self.t, qf_loss.mean().item(), policy_loss.item()]
            )

    def after_learn_on_batch(self):
        # update target network
        if self.t % self.args.target_update_freq == 0:
            ptu.soft_update_from_to(
                self.q_learner, self.q_target, self.args.tau
            )
            ptu.soft_update_from_to(
                self.policy, self.policy_target, self.args.tau
            )

    def state_value_pred(self, s):
        """
        log the state value of initial state
            s: (s_dim, )
        """
        with torch.no_grad():
            s = s.unsqueeze(dim=0)  # (1, s_dim)
            action = self.policy(s)  # (1, a)
            target_q_values = self.q_target(s, action) # (1, 1)
        return torch.unsqueeze(target_q_values, dim=0).item()
