import random

import numpy as np


class ReplayBuffer():
    def __init__(self, input_shape, batch_size, capacity, alpha):


        self.batch_size = batch_size
        self.capacity = capacity # bound memory so we don't crash RAM
        self.alpha = alpha
        self.mem_cntr = 0 # simulate stack by knowing whee in memory we are

        self.priority_sum = [0 for _ in range(2*capacity)]
        self.priority_min = [float('inf') for _ in range(2 * capacity)]
        self.max_priority = 1.

        self.data = {
            'state': np.zeros((self.capacity, *input_shape), dtype=np.float32),
            'action': np.zeros(self.capacity, dtype=np.int64),
            'reward': np.zeros(self.capacity, dtype=np.float32),
            'state_': np.zeros((self.capacity, *input_shape), dtype=np.float32),
            'done':np.zeros(self.capacity, dtype=bool)
        }

        self.next_idx = 0
        self.size = 0

    def add(self, state, state_, reward, action, done):
        self.mem_cntr += 1
        idx = self.next_idx

        self.data['state'][idx] = state
        self.data['state_'][idx] = state_
        self.data['reward'][idx] = reward
        self.data['action'][idx] = action
        self.data['done'][idx] = done

        self.next_idx = (idx + 1) % self.capacity

        self.size = min(self.capacity, self.size+1)

        priority_alpha = self.max_priority ** self.alpha

        self._set_piority_min(idx, priority_alpha)
        self._set_priority_sum(idx, priority_alpha)

    def _set_piority_min(self, idx, priority_alpha):
        idx += self.capacity
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx //= 2

            self.priority_min[idx] = min(self.priority_min[2*idx],
                                         self.priority_min[2*idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.capacity

        self.priority_sum[idx] = priority

        while idx >= 2:
            idx//=2
            self.priority_sum[idx] = self.priority_sum[2*idx] +\
                                        self.priority_sum[2*idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.capacity:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2* idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2*idx + 1

        return idx - self.capacity

    def sample(self, beta):
        samples = {
            'weights': np.zeros(self.batch_size, dtype=np.float32),
            'indexes': np.zeros(self.batch_size, dtype=np.int32)
        }
        for i in range(self.batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            samples['indexes'][i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size) ** (-beta)
        for i in range(self.batch_size):
            idx = samples['indexes'][i]
            prob = self.priority_sum[idx + self.capacity] / self._sum()
            weight = (prob * self.size) ** (-beta)

            samples['weights'][i] = weight / max_weight

        for k,v in self.data.items():
            samples[k] = v[samples['indexes']]
        return samples

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = priority ** self.alpha

            self._set_piority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def is_sufficient(self):
        return self.size > self.batch_size


from DDQN.classes.sumtree import SumTree


class PrioritizedBuffer:

    def __init__(self, max_size, batch_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.current_length = 0
        self.batch_size = batch_size


    def add(self, state, next_state, reward,  action, done):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length = self.current_length + 1
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience)

    def sample(self):
        batch_idx, batch, IS_weights = [], [], []
        segment = self.sum_tree.total() / self.batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)

            batch_idx.append(idx)
            batch.append(data)
            prob = p / p_sum
            IS_weight = (self.sum_tree.total() * prob) ** (-self.beta)
            IS_weights.append(IS_weight)

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return state_batch, action_batch, np.array(reward_batch), next_state_batch, done_batch, IS_weights, batch_idx

    def update_priorities(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def is_sufficient(self):
        return self.current_length > self.batch_size