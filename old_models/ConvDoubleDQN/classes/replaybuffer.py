import random

import numpy as np

from old_models.ConvDoubleDQN.classes.sumtree import SumTree


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



class RandomBuffer():
    def __init__(self, input_shape, max_size, batch_size, beta, multi_frame:bool=True):
        self.beta = beta

        self.multi_frame = multi_frame

        self.batch_size = batch_size
        self.mem_size = max_size # bound memory so we don't crash RAM
        self.mem_cntr = 0 # simulate stack by knowing whee in memory we are

        input_shape = (input_shape[0], input_shape[1], input_shape[2])

        # initialise state memory with 0 values
        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_mem_ = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)

    def add(self, state, state_, reward,  action, done):
        # wrap counter to know where we are in memory
        idx = self.mem_cntr % self.mem_size

        if self.multi_frame == True:
            # state = transforms.ToPILImage()(state)
            state = state.numpy()
            state_ = state_.numpy()

        # assign states, reward and action to memory
        self.state_mem[idx] = state
        self.state_mem_[idx] = state_
        self.reward_mem[idx] = reward
        self.action_mem[idx] = action
        self.terminal_mem[idx] = done
        # increment position in memory
        self.mem_cntr += 1

    def sample(self):
        # Max existing memory size (if mem not full set max mem to counter value)
        max_mem = min(self.mem_size, self.mem_cntr)
        btch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_mem[btch]
        states_ = self.state_mem_[btch]
        actions = self.action_mem[btch]
        rewards = self.reward_mem[btch]
        terminal = self.terminal_mem[btch]

        return np.array(states), actions, np.array(rewards), np.array(states_), terminal

    def is_sufficient(self):
        return self.mem_cntr > self.batch_size