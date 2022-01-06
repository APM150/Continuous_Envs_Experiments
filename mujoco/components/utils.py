import random
import numpy as np
import torch
import datetime


def set_seed(seed):

    """ ensure full reproducibility """

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def param_space_diff(param1, param2):
    """ compute L2^2 distance between parameters """
    with torch.no_grad():
        params = []
        for p1, p2 in zip(param1, param2):
            params.append((p1 - p2) ** 2)
        for i in range(len(params)):
            params[i] = params[i].view(-1)
        return torch.cat(params, dim=0).sum().item()

def now() -> str:
    return datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")


def bisect1d(f, a, b, init, threashold, size, maxiter, device) -> (torch.Tensor, torch.Tensor):
    """ bisect for 1d torch.tensor
        return  x
                err     error value for debug
    """
    low = torch.ones(size=(size, )).float().to(device) * a
    high = torch.ones(size=(size,)).float().to(device) * b
    x = torch.ones(size=(size, )).float().to(device) * (init)
    y = f(x)
    # handle out of bound values
    leftout = f(low) > 0
    rightout = f(high) < 0
    x[leftout] = low[leftout]
    x[rightout] = high[rightout]
    search_region = torch.logical_and(~leftout, ~rightout)
    zero = torch.zeros((torch.sum(search_region).item(),)).float().to(device)
    # binary search
    i = 0
    while not torch.allclose(y[search_region], zero, atol=threashold):
        moveleft = y > 0
        moveright = y < 0
        high[moveleft] = x[moveleft]
        x[moveleft] = (low[moveleft] + x[moveleft]) / 2
        low[moveright] = x[moveright]
        x[moveright] = (high[moveright] + x[moveright]) / 2
        y = f(x)
        i += 1
        if maxiter is not None and i == maxiter:
            break
    err = y[search_region] - zero
    return x, err

class OUProcess:
    def __init__(
            self,
            action_space,
            mu=0,
            theta=0.15,
            max_sigma=0.2,
            min_sigma=None,
            decay_period=100000,
    ):
        if min_sigma is None:
            min_sigma = max_sigma
        self.sigma = max_sigma
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self.theta = theta
        self.mu = mu
        self.state = torch.ones(action_space) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

    def get_action_from_raw_action(self, action, t=0, **kwargs):
        ou_state = self.evolve_state()
        self.sigma = (
            self._max_sigma
            - (self._max_sigma - self._min_sigma)
            * min(1.0, t * 1.0 / self._decay_period)
        )
        return np.clip(action + ou_state, -1, 1)



"""
    rlkit utils
"""
"""
Common evaluation utilities.
"""

from collections import OrderedDict
from numbers import Number

import numpy as np
import collections

def list_of_dicts__to__dict_of_lists(lst):
    """
    ```
    x = [
        {'foo': 3, 'bar': 1},
        {'foo': 4, 'bar': 2},
        {'foo': 5, 'bar': 3},
    ]
    ppp.list_of_dicts__to__dict_of_lists(x)
    # Output:
    # {'foo': [3, 4, 5], 'bar': [1, 2, 3]}
    ```
    """
    if len(lst) == 0:
        return {}
    keys = lst[0].keys()
    output_dict = collections.defaultdict(list)
    for d in lst:
        assert set(d.keys()) == set(keys), (d.keys(), keys)
        for k in keys:
            output_dict[k].append(d[k])
    return output_dict


def get_generic_path_information(paths, stat_prefix=''):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    actions = [path["actions"] for path in paths]
    if len(actions[0].shape) == 1:
        actions = np.hstack([path["actions"] for path in paths])
    else:
        actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)
    statistics[stat_prefix + 'Average Returns'] = get_average_returns(paths)

    for info_key in ['env_infos', 'agent_infos']:
        if info_key in paths[0]:
            all_env_infos = [
                ppp.list_of_dicts__to__dict_of_lists(p[info_key])
                for p in paths
            ]
            for k in all_env_infos[0].keys():
                final_ks = np.array([info[k][-1] for info in all_env_infos])
                first_ks = np.array([info[k][0] for info in all_env_infos])
                all_ks = np.concatenate([info[k] for info in all_env_infos])
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    final_ks,
                    stat_prefix='{}/final/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    first_ks,
                    stat_prefix='{}/initial/'.format(info_key),
                ))
                statistics.update(create_stats_ordered_dict(
                    stat_prefix + k,
                    all_ks,
                    stat_prefix='{}/'.format(info_key),
                ))

    return statistics


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=True,
        exclude_max_min=False,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


if __name__ == '__main__':
    def f(x):
        return x**2 + 8*x - 5

    print(bisect1d(f, -20, 20, 10, 1e-5, 1, 10, 'cpu'))
