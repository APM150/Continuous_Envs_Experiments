import torch
import torch.nn as nn

from components.distributions import TanhNormal
from models.ensemble import EnsembleLinear

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class TanhMlpPolicy(nn.Module):

    def __init__(self, args, obs_dim, action_dim):
        super(TanhMlpPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=args.network_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=args.network_size[0], out_features=args.network_size[1]),
            nn.ReLU()
        )
        self.last_layer = nn.Linear(in_features=args.network_size[1], out_features=action_dim)
        self.last_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.last_layer.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        h = self.fc(obs)
        action = self.last_layer(h)
        action = torch.tanh(action)
        return action


class TanhGaussianPolicy(nn.Module):

    def __init__(self, args, obs_dim, action_dim):
        super(TanhGaussianPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=obs_dim, out_features=args.network_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=args.network_size[0], out_features=args.network_size[1]),
            nn.ReLU()
        )
        self.mu = nn.Linear(in_features=args.network_size[1], out_features=action_dim)
        self.sigma = nn.Linear(in_features=args.network_size[1], out_features=action_dim)
        self.sigma.weight.data.uniform_(-1e-3, 1e-3)
        self.sigma.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs):
        h = self.fc(obs)
        mean = self.mu(h)
        log_std = self.sigma(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)


class Ensemble_Policy_Network(nn.Module):

    def __init__(self, args, obs_dim, action_dim):
        super(Ensemble_Policy_Network, self).__init__()
        self.ensemble_size = args.ensemble_size
        self.fc = nn.Sequential(
            EnsembleLinear(self.ensemble_size, obs_dim, args.network_size[0]),
            nn.ReLU(),
            EnsembleLinear(self.ensemble_size, args.network_size[0], args.network_size[1]),
            nn.ReLU()
        )
        self.mu = EnsembleLinear(self.ensemble_size, args.network_size[1], action_dim)
        self.sigma = EnsembleLinear(self.ensemble_size, args.network_size[1], action_dim)
        self.sigma.weight.data.uniform_(-1e-3, 1e-3)
        self.sigma.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs):
        h = self.fc(obs)
        mean = self.mu(h)
        log_std = self.sigma(h)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)
        return TanhNormal(mean, std)


if __name__ == "__main__":
    import components.pytorch_util as ptu
    from types import SimpleNamespace
    import time
    import numpy as np
    import torch
    import gym

    args = SimpleNamespace(

        # env
        device='cpu',
        seed=1234,
        max_episode_length=int(108e3),
        env='HalfCheetah-v2',
        history_length=1,

        # UQL
        ensemble_size=5,

        # mem
        capacity=500000,
        beta_mean=1,
        num_ensemble=3,
        discount=0.99,
        multi_step=1,
        priority_weight=0.4,
        priority_exponent=0.5,
        network_size=[400, 300],

        # training
        learn_start=1600,
        num_epochs=50,
        num_steps_per_epoch=10_000,
        min_num_steps_before_training=1600,

        resampling_rate=None,
    )
    ptu.set_gpu_mode(True)
    # policy = TanhGaussianPolicy(17, 6).to('cuda')
    policy = TanhMlpPolicy(args, 17, 6).to('cuda')
    obs = torch.zeros(17).to('cuda')
    # dist = policy(obs)
    # print(dist.rsample_and_logprob())
    print(policy(obs))
