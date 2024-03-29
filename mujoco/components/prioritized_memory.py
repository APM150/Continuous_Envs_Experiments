# -*- coding: utf-8 -*-
from __future__ import division

import code
from collections import namedtuple
import numpy as np
import torch
import gym

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))


# Wrap-around cyclic buffer
class CyclicBuffer:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer

    def append(self, data):
        self.data[self.index] = data  # Store data in underlying data structure
        self.index = (self.index + 1) % self.size  # Update index

    def get(self, data_index):
        return self.data[data_index % self.size]


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree:
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1, ), dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
        # self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    # def append(self, data, value):
    def append(self, value):
        # self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    # def get(self, data_index):
    #     return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

class ReplaySampler:
    def __init__(self, args, capacity, data_buffer=None):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.priorities = SegmentTree(capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities
        self.data = data_buffer

        self.blank_trans = Transition(0, torch.zeros(gym.make(args.env).observation_space.shape[0], dtype=torch.float32), None, 0, False)

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, action, reward, terminal):
        # state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device('cpu'))  # Only store last frame and discretise to save memory
        # self.priorities.append(Transition(self.t, state, action, reward, not terminal), self.priorities.max)  # Store new transition with maximum priority
        self.priorities.append(self.priorities.max)
        self.t = 0 if terminal else self.t + 1  # Start new episodes with t = 0

    # Returns a transition with blank states where appropriate
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.data.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                transition[t] = self.blank_trans  # If future frame has timestep 0
            else:
                transition[t] = self.data.get(idx - self.history + 1 + t)
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.data.get(idx - self.history + 1 + t)
            else:
                transition[t] = self.blank_trans  # If prev (next) frame is terminal
        return transition

    # Returns a valid sample from a segment
    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment, (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.priorities.find(sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.priorities.index - idx) % self.capacity > self.n \
                    and (idx - self.priorities.index) % self.capacity >= self.history \
                    and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.history]]).squeeze().to(dtype=torch.float32, device=self.device)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).squeeze().to(dtype=torch.float32, device=self.device)
        # Discrete action to be used as index
        action = transition[self.history - 1].action.to(dtype=torch.float32, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.discount ** n * transition[self.history + n - 1].reward for n in range(self.n))], dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32, device=self.device)
        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.priorities.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.stack(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.priorities.full else self.priorities.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, 1 - nonterminals, weights


    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.priorities.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.data.data[self.current_idx].state
        prev_timestep = self.data.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = self.blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.data.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device)  # Agent will turn into batch
        self.current_idx += 1
        return state

    next = __next__  # Alias __next__ for Python 2 compatibility


class ReplayMemory:

    def __init__(self, args):
        self.args = args
        self.data_buffer = CyclicBuffer(args.capacity)
        self.sampler = ReplaySampler(args, args.capacity, data_buffer=self.data_buffer)
        total_ts = self.args.num_epochs * self.args.num_steps_per_epoch
        self.weight_increase = (1 - self.args.priority_weight) / total_ts

    def append(self, state, action, reward, terminal):
        state = state.to(device=torch.device('cpu'))
        self.data_buffer.append(Transition(self.sampler.t, state, action, reward, not terminal))
        self.sampler.append(state, action, reward, terminal)

    def sample(self, batch_size):
        return self.sampler.sample(batch_size)

    def anneal_priority_weight(self):
        self.sampler.priority_weight = min(self.sampler.priority_weight + self.weight_increase, 1)

    def update_priorities(self, idxs, priorities):
        self.sampler.update_priorities(idxs, priorities)


class EnsembleReplayMemory:

    def __init__(self, args):
        self.args = args
        self.data_buffer = CyclicBuffer(args.capacity)
        self.samplers = [ReplaySampler(args, args.capacity, data_buffer=self.data_buffer)
                         for _ in range(args.ensemble_size)]
        total_ts = self.args.num_epochs * self.args.num_steps_per_epoch
        self.weight_increase = (1 - self.args.priority_weight) / (total_ts - args.min_num_steps_before_training)

    def append(self, state, action, reward, terminal):
        state = state.to(device=torch.device('cpu'))
        self.data_buffer.append(Transition(self.samplers[0].t, state, action, reward, not terminal))
        for k in range(self.args.ensemble_size):
            self.samplers[k].append(state, action, reward, terminal)

    def sample(self, batch_size, learners):
        """ returns: idxs, states, actions, returns, next_states, dones, weights """
        idxs_k, states_k, actions_k, returns_k, next_states_k, nonterminals_k, weights_k = [], [], [], [], [], [], []
        for k in learners:
            idxs, states, actions, returns, next_states, nonterminals, weights = self.samplers[k].sample(batch_size)
            idxs_k.append(idxs)
            states_k.append(torch.unsqueeze(states, dim=1))
            actions_k.append(torch.unsqueeze(actions, dim=1))
            returns_k.append(returns.view(batch_size, 1, 1))
            next_states_k.append(torch.unsqueeze(next_states, dim=1))
            nonterminals_k.append(torch.unsqueeze(nonterminals, dim=1))
            weights_k.append(weights.view(batch_size, 1))
        states = torch.cat(states_k, dim=1)
        actions = torch.cat(actions_k, dim=1)
        returns = torch.cat(returns_k, dim=1)
        next_states = torch.cat(next_states_k, dim=1)
        dones = torch.cat(nonterminals_k, dim=1)
        weights = torch.cat(weights_k, dim=1)
        return idxs_k, states, actions, returns, next_states, dones, weights

    def anneal_priority_weight(self):
        for k in range(self.args.ensemble_size):
            self.samplers[k].priority_weight = min(self.samplers[k].priority_weight + self.weight_increase, 1)

    def update_priorities(self, idxs, priorities, learners):
        for i, k in enumerate(learners):
            self.samplers[k].update_priorities(idxs[i], priorities[i])

if __name__ == '__main__':
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

        # training
        learn_start=1600,
        num_epochs=50,
        num_steps_per_epoch=10_000,
        min_num_steps_before_training=1600,
    )

    env = gym.make(args.env)
    # env.eval()
    # mem = EnsembleReplayMemory(args)
    mem = ReplayMemory(args)

    done = False
    R = 0
    s = env.reset()
    t = 0
    # print(s.shape)
    while t < args.learn_start:
        if done:
            R = 0
            s = env.reset()
        s = torch.from_numpy(s)
        a = torch.from_numpy(np.random.random(6))
        s_next, r, done, _ = env.step(a)
        R += r
        r_clip = max(min(r, 1), -1)
        mem.append(s, a, r_clip, done)
        s = s_next
        t += 1
    print(t)

    mem.anneal_priority_weight()
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(2)
    # idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(2, [0,1,2])
    print(nonterminals.shape)
