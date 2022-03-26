import numpy as np
import torch as T
#from classes.replaybuffer import ReplayBuffer # Simple Replay Buffer
from classes.replaybuffer_ import ReplayBuffer
from classes.ddqn import DDQN

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='networkdata/', name='maze-test-1.pt', combined=False):
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

        self.memory = ReplayBuffer(input_dims, mem_size, batch_size, combined)

        self.q_eval = DDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                           name=name, save_dir=self.save_dir)
        self.q_next = DDQN(self.lr, self.n_actions, input_dims=self.input_dims,
                           name=name, save_dir=self.save_dir)

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
        if not self.memory.is_sufficient():
            return

        # Start AD
        self.q_eval.optimiser.zero_grad()
        self.replace_target_network()

        # sample memory
        state, actions, reward, state_, term = \
                self.memory.sample_buffer()

        states = T.tensor(state).to(self.q_eval.device)
        #actions = T.tensor(action).to(self.q_eval.device)
        term = T.tensor(term).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)

        idxs = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[idxs, actions]

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