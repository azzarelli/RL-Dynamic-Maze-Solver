
import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer():
    def __init__(self, input_shape, max_size, batch_size):
        self.batch_size = batch_size
        self.mem_size = max_size # bound memory so we don't crash RAM
        self.mem_cntr = 0 # simulate stack by knowing whee in memory we are

        # initialise state memory with 0 values
        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_mem_ = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)

    def store_buffer(self, state, state_, reward, action, done):
        # wrap counter to know where we are in memory
        idx = self.mem_cntr % self.mem_size
        # assign states, reward and action to memory
        self.state_mem[idx] = state
        self.state_mem_[idx] = state_
        self.reward_mem[idx] = reward
        self.action_mem[idx] = action
        self.terminal_mem[idx] = done
        # increment position in memory
        self.mem_cntr += 1

    def sample_buffer(self):
        # Max existing memory size (if mem not full set max mem to counter value)
        max_mem = min(self.mem_size, self.mem_cntr)
        btch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_mem[btch]
        states_ = self.state_mem_[btch]
        actions = self.action_mem[btch]
        rewards = self.reward_mem[btch]
        terminal = self.terminal_mem[btch]

        return states, actions, rewards, states_, terminal

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size