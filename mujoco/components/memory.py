import torch
import numpy as np
import gym


class ReplayMemory:

    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env)
        self.obs_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        self.mem_size = args.capacity
        self.mem_cntr = 0
        self.state_memory = torch.zeros((self.mem_size, self.obs_space), dtype=torch.float32)
        self.new_state_memory = torch.zeros((self.mem_size, self.obs_space), dtype=torch.float32)
        self.action_memory = torch.zeros((self.mem_size, self.action_space), dtype=torch.float32)
        self.reward_memory = torch.zeros(self.mem_size)
        self.terminal_memory = torch.zeros(self.mem_size, dtype=torch.float32)

    def append(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        """
        Output:
            s: (b, state_dim)
            a: (b, action_dim)
            r: (b, 1)
            s_:(b, state_dim)
            d: (b, 1)
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch].to(self.args.device)
        states_ = self.new_state_memory[batch].to(self.args.device)
        actions = self.action_memory[batch].to(self.args.device)
        rewards = self.reward_memory[batch].to(self.args.device)
        dones = self.terminal_memory[batch].to(self.args.device)

        return (
            states,
            actions,
            rewards,
            states_,
            dones.unsqueeze(-1)
        )

    def __len__(self):
        return min(self.mem_cntr, self.mem_size)


class EnsembleReplayMemory:

    def __init__(self, args):
        self.args = args
        self.ensemble_size = args.ensemble_size
        self.resampling_rate = args.resampling_rate
        self.main_buffer = ReplayMemory(args)
        if self.resampling_rate is not None:
            self.ensemble_buffers = [ReplayMemory(args) for _ in range(self.ensemble_size)]

    def __len__(self):
        return len(self.main_buffer)

    def append(self, state, action, reward, state_, done):
        self.main_buffer.append(state, action, reward, state_, done)
        if self.resampling_rate is None:
            return
        if self.resampling_rate != 0:
            number_of_duplicates = np.random.poisson(lam=self.resampling_rate, size=len(self.ensemble_buffers))
        else:
            number_of_duplicates = np.zeros((len(self.ensemble_buffers),), dtype=np.int)
            number_of_duplicates[np.random.randint(len(self.ensemble_buffers))] = 1
        if np.sum(number_of_duplicates) == 0:  # don't throw away any data
            selected = np.random.randint(len(self.ensemble_buffers))
            self.ensemble_buffers[selected].append(state, action, reward, state_, done)
        else:  # stream in number of sample follow N ~ poisson(lambda=resampling_rate)
            for k in range(len(self.ensemble_buffers)):
                for _ in range(number_of_duplicates[k]):
                    self.ensemble_buffers[k].append(state, action, reward, state_, done)

    def sample(self, batch_size):
        """
        Output:
            s: (b, k, state_dim)
            a: (b, k, action_dim)
            r: (b, k)
            s_:(b, k, state_dim)
            d: (b, k, 1)
        """
        if self.resampling_rate is None:
            batches = [self.main_buffer.sample(batch_size) for _ in range(self.ensemble_size)]
        else:
            batches = [buffer.sample(batch_size) for buffer in self.ensemble_buffers]
        res = []
        for item in range(5):
            transpose10 = list(range(1, len(batches[0][item].shape) + 1))
            transpose10.insert(1, 0)
            ensemble_batch = torch.cat([torch.unsqueeze(batch[item], dim=0) for batch in batches], dim=0)
            ensemble_batch = ensemble_batch.permute(*transpose10)
            res.append(ensemble_batch)
        return tuple(res)

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

        resampling_rate=None,
    )

    env = gym.make(args.env)
    mem = EnsembleReplayMemory(args)
    # mem = ReplayMemory(args)

    done = False
    R = 0
    s = env.reset()
    s = torch.from_numpy(s)
    t = 0
    # print(s.shape)
    while t < args.learn_start:
        if done:
            R = 0
            s = env.reset()
            s = torch.from_numpy(s)
        a = torch.from_numpy(np.random.random(6))
        s_next, r, done, _ = env.step(a)
        s_next = torch.from_numpy(s_next)
        R += r
        r_clip = max(min(r, 1), -1)
        mem.append(s, a, r_clip, s_next, done)
        s = s_next
        t += 1
    print(t)

    # states, actions, returns, next_states, nonterminals = mem.sample(2)
    states, actions, returns, next_states, nonterminals = mem.sample(2)
    print(returns.shape)
