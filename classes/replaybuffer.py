import random

import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

    def is_full(self):
        return self.capacity == self.size

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size