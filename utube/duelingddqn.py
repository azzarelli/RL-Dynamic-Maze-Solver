import time

import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayBuffer():
    def __init__(self, max_size, input_shape):
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

    def sample_buffer(self, batch_size):
        # Max existing memory size (if mem not full set max mem to counter value)
        max_mem = min(self.mem_size, self.mem_cntr)
        btch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[btch]
        states_ = self.state_mem_[btch]
        actions = self.action_mem[btch]
        rewards = self.reward_mem[btch]
        terminal = self.terminal_mem[btch]

        return states, actions, rewards, states_, terminal

class DDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, save_dir):
        super(DDQN, self).__init__()
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, name)

        self.fc1 = nn.Linear(*input_dims, 512)
        self.V = nn.Linear(512,1)
        self.A = nn.Linear(512, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'Training on {self.device} ...')
        self.to(self.device)

    def forward(self, state):
        output = self.fc1(state)
        relu = F.relu(output)
        V = self.V(relu)
        A = self.A(relu)

        return V,A

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='/savedirectory'):
        self.learn_step_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_thresh = replace
        self.save_dir = save_dir

        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                           name='maze-test-1', save_dir=self.save_dir)
        self.q_next = DDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                           name='maze-test-1', save_dir=self.save_dir)

    def greedy_epsilon(self, observation):
        # if we randomly choose max expected reward action
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        # otherwise random action
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, state_, reward, action, done):
        self.memory.store_buffer(state, state_, reward, action, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_thresh == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def dec_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_()
        self.q_next.save_()

    def load_models(self):
        self.q_eval.load_save()
        self.q_next.load_save()

    def learn(self):
        # Wait for memory to fill up before learning from empty set
        if self.memory.mem_cntr < self.batch_size:
            return

        # Start AD
        self.q_eval.optimiser.zero_grad()
        self.replace_target_network()

        # sample memory
        state, action, reward, state_, term = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        term = T.tensor(term).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)

        idxs = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(  V_s, ( A_s - A_s.mean(dim=1, keepdim=True) )  )[idxs, actions]

        q_next = T.add(V_s_,(A_s_ - A_s.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        # apply mask for terminates networks
        q_next[term] = 0.0

        q_target = rewards + self.gamma*q_next[idxs, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimiser.step()
        self.learn_step_counter += 1

        self.dec_epsilon()

from lib.read_maze import get_local_maze_information
class Environment():

    def __init__(self):
        self.step_cntr = 0
        self.actor_pos = (1, 1)
        self.observation = self.observe_environment  # set up empty state

    @property
    def reset(self):
        self.step_cntr = 0
        self.actor_pos = (1, 1)
        self.observation = self.observe_environment  # set up empty state
        return self.observation

    @property
    def observe_environment(self):
        loc = get_local_maze_information(*self.actor_pos)
        loc_vec = []
        for l in loc:
            for j in l:
                loc_vec.append(j[0]) # index % 2 in loc_vec = wall data
                loc_vec.append(j[1]) # index % 3 (and index 1) in loc_vec = fire data
        return loc_vec

    @property
    def get_local_matrix(self):
        """Return observation matrix from 1-D observation data

        Notes
        -----
        Observation is a 1-Dim vector with wall and fire data in alternating index values
        self.observation = [w_00, f_00, w_10, f_10, ..., w_2_2, f_22] for w_rc, f_rc, where r - rows and c - cols
        (note r, c is called as `obj[r][c]` in class-method `step`)

        We return the observation matrix rather than 1-D array to facilitate calling the observation within the environment
        i.e. accessing through `[r][c]` is much easier than cycling through `obj[i]`
        """
        loc_mat = [[[0,0] for i in range(3) ]for j in range(3)]
        for i in range(len(self.observation)):
            if i % 2 == 0: # even index values refer to wall-data, index obj[r][c][0]
                # index values to expect - 0, 2, 4 | 6, 8, 10 | 12, 14, 16
                col = (i % 6) / 2 # gives position 0,1,2
                if i < 5: row = 0
                elif i < 11: row = 1
                else: row = 2

                loc_mat[row][int(col)][0] = self.observation[i]

            else:  # odd index values refer to wall-data, index obj[r][c][1]
                # index values to expect - 1, 3, 5 | 7, 9, 11 | 13, 15, 17
                col = ((i-1) % 6) / 2  # gives position 0,1,2
                if i < 6:
                    row = 0
                elif i < 12:
                    row = 1
                else:
                    row = 2

                loc_mat[row][int(col)][1] = self.observation[i]

        return loc_mat

    @property
    def get_actor_pos(self):
        return self.actor_pos

    def step(self, action):
        """Sample environment dependant on action which has occurred

        Action Space
        ------------
        0 - no move
        1 - up
        2 - left
        3 - down
        4 - right

        Environment Rules
        -----------------
        - if we try to walk into a wall our character stays fixed, v small penalty for simply not choosing to stay
        - if we walk into a fire, the game is terminated and we give a penalty
        - if we take a step away fom actors prior position (maybe ref 1,1, or actual prior pos) reward,
        - however if we take a step back from end point, reward = 0
        - if we reach 199*199 we receive a reward of `R` (dependant on the number of steps it took to get there)

        - TODO - Later we could give rewards for not moving when all paths are blocked by fires
        """
        time.sleep(1)

        self.step_cntr += 1
        reward = 0
        done = 0
        x_loc, y_loc = 0, 0 # initialise local positions within observation array

        # First check if actor wants to move
        if action == 0:
            return self.observe_environment, reward, done, {}

        x, y = self.actor_pos

        # Horizontal movement, determine which position we move into (both local and global pos reference)
        if action % 2 == 0:
            x_loc = 1 + (action - 3)
            new_pos = (x + (action - 3) ,y) # left: x + 2-3 = x - 1; right: x + 4-3 = x + 1
        # Vertical Movement
        else:
            y_loc = 1 - (action - 2)
            new_pos = (x, y + (action - 2)) # up: y + 1-2 = y + 1; down: y + 3-2 = y + 1

        observation_mat = self.get_local_matrix
        self.actor_pos = new_pos

        # Check for wall in way
        if observation_mat[y_loc][x_loc][0] == 0:
            reward -= 1
            return self.observe_environment, reward, done, {}

        # Check for fire
        elif observation_mat[y_loc][x_loc][1] > 0:
            # TODO - fire dies out `observation[x][y][1]` steps later, so maybe remeber this somehow?
            reward = -100
            done = 1
            return self.observe_environment, reward, done, {}

        # Successful Move into empty area
        else:
            self.actor_pos = new_pos # set new position to observe new state-space
            # We have moved away from area
            if x_loc > 1 or y_loc > 1:
                reward = 5 # TODO - too high/low idk?
                if new_pos == (199, 199):
                    reward = 100 # TODO - Make this vary with step counter, so diminishing returns if take too long
            else:
                reward = 0
            return self.observe_environment, reward, done, {}





