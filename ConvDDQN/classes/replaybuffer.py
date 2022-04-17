import random

import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque
from classes.sumtree import SumTree
from torchvision.transforms import transforms

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

        # state = transforms.ToPILImage()(state)
        img = state.numpy()
        state = img # T.from_numpy(img[0])
        img_ = next_state.numpy()
        next_state = img_
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

        return np.array(state_batch), action_batch, np.array(reward_batch), np.array(next_state_batch), done_batch, IS_weights, batch_idx

    def update_priorities(self, idx, td_error):
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority)

    def is_sufficient(self):
        return self.current_length > self.batch_size