import torch
import torch.nn as nn

from models.ensemble import EnsembleLinear


class Q_Network(nn.Module):
#TODO: optimize the network size
    def __init__(self, args, obs_dim, action_dim):
        super(Q_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=obs_dim+action_dim, out_features=args.network_size[0]),
            nn.ReLU(),
            nn.Linear(in_features=args.network_size[0], out_features=args.network_size[1]),
            nn.ReLU(),
        )
        self.last_fc = nn.Linear(in_features=args.network_size[1], out_features=1)
        self.last_fc.weight.data.uniform_(-3e-3, 3e-3)
        self.last_fc.bias.data.fill_(0)

    def forward(self, obs, action):
        flat_inputs = torch.cat([obs, action], dim=1)
        h = self.fc(flat_inputs)
        return self.last_fc(h)


class Ensemble_Q_Network(nn.Module):

    def __init__(self, args, obs_dim, action_dim):
        super(Ensemble_Q_Network, self).__init__()
        self.ensemble_size = args.ensemble_size
        self.fc = nn.Sequential(
            EnsembleLinear(self.ensemble_size, obs_dim+action_dim, args.network_size[0]),
            nn.ReLU(),
            EnsembleLinear(self.ensemble_size, args.network_size[0], args.network_size[1]),
            nn.ReLU(),
        )
        self.last_fc = EnsembleLinear(self.ensemble_size, args.network_size[1], 1)
        self.last_fc.weight.data.uniform_(-3e-3, 3e-3)
        self.last_fc.bias.data.fill_(0)

    def forward(self, obs, action):
        """
        obs: (b, k, obs_dim)
        action: (b, k, action_dim)
        """
        flat_inputs = torch.cat([obs, action], dim=-1)
        h = self.fc(flat_inputs)
        return self.last_fc(h)
