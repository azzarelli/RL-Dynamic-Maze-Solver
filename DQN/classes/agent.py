import numpy as np
import torch as T
from classes.replaybuffer import ReplayBuffer # Simple Replay Buffer
#from classes.replaybuffer_ import ReplayBuffer
#from DQN.classes.memory.per import PrioritizedReplayBuffer

from DQN.classes.dqn import DQN

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=10, save_dir='networkdata/', name='default'):
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

        self.memory = ReplayBuffer(input_dims, mem_size, batch_size)

        self.q_eval = DQN(self.lr, self.n_actions, input_dims=input_dims,
                           name=name, save_dir=self.save_dir)
        self.q_next = DQN(self.lr, self.n_actions, input_dims=input_dims,
                           name=name+'.next', save_dir=self.save_dir)



    def greedy_epsilon(self, observation):
        actions = []
        # if we randomly choose max expected reward action
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        # otherwise random action
        else:
            action = np.random.choice(self.action_space)
        return action, actions

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
        if not self.memory.is_sufficient():
            return

        # Start AD
        self.q_eval.optimiser.zero_grad()
        self.replace_target_network()

        # sample memory
        state, actions, reward, state_, term = \
                self.memory.sample_buffer()
        idxs = np.arange(self.batch_size) # need for array slicing later

        state_btch = T.tensor(state).to(self.q_eval.device)
        term_btch = T.tensor(term).to(self.q_eval.device)
        rewards_btch = T.tensor(reward).to(self.q_eval.device)
        state_btch_ = T.tensor(state_).to(self.q_eval.device)



        q_pred = self.q_eval.forward(state_btch)[idxs, actions]
        q_next = self.q_next.forward(state_btch_)
        q_eval = self.q_eval.forward(state_btch_)

        max_actions = T.argmax(q_eval, dim=1)

        # apply mask for terminates networks
        q_next[term_btch] = 0.0

        q_target = rewards_btch + self.gamma * q_next[idxs, max_actions]

        print(q_target, q_pred)
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)

        loss.backward()
        self.q_eval.optimiser.step()
        self.learn_step_counter += 1
        self.dec_epsilon()

        return loss.item()
