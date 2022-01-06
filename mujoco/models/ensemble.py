import math
import torch
import torch.nn as nn


class EnsembleLinear(nn.Module):

    """ linear layer optimized for ensemble """

    def __init__(self, ensemble_size, input_size, output_size):
        super(EnsembleLinear, self).__init__()
        self.weight = nn.Parameter(torch.zeros((ensemble_size, input_size, output_size)))
        self.bias = nn.Parameter(torch.zeros((ensemble_size, output_size)))

        for i in range(ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x):
        """ x (b, k, a) """
        x_ind = 'bki' if len(x.shape) == 3 else 'bi'
        return torch.einsum(f'{x_ind},kio->bko', x, self.weight) + self.bias
