import numpy as np
import torch as T
from classes.replaybuffer import ReplayBuffer # Simple Replay Buffer
#from classes.replaybuffer_ import ReplayBuffer

from classes.ddqn import DDQN

def TD_errors(q_pred, q_tru):
    return q_pred - q_tru

def get_priorities(q_pred, q_tru, err=1e-6):
    with T.no_grad():
        return np.abs(TD_errors(q_pred, q_tru).cpu().numpy()) + err

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='networkdata/', name='maze-test-1.pt'):
        self.learn_step_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size

        self.replace_target_thresh = replace
        self.save_dir = save_dir

        self.action_space = [i for i in range(self.n_actions)]

        self.memory = ReplayBuffer(input_dims,  batch_size, mem_size, 0.05)

        self.q_eval = DDQN(self.lr, self.n_actions, input_dims=input_dims,
                           name=name, save_dir=self.save_dir)
        self.q_next = DDQN(self.lr, self.n_actions, input_dims=input_dims,
                           name=name+'.next', save_dir=self.save_dir)



    def greedy_epsilon(self, observation):
        with T.no_grad():
            actions = []
            # if we randomly choose max expected reward action
            if np.random.random() > self.epsilon:
                state = T.tensor(observation, dtype=T.float).to(self.q_eval.device)
                _, advantage = self.q_eval.forward(state)
                #print(advantage, _, T.argmax(advantage).item())
                action = T.argmax(advantage).item()
                actions = advantage
            # otherwise random action
            else:
                action = np.random.choice(self.action_space)
            return action, actions

    def store_transition(self, state, state_, reward, action, done):
        self.memory.add(state, state_, reward, action, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_thresh == 0:
            print('Replacing Target')
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
        self.replace_target_network()
        self.q_eval.optimiser.zero_grad()

        # sample memory
        sample = self.memory.sample(0.4)
        state = sample['state']
        state_ = sample['state_']
        actions = sample['action']
        reward = sample['reward']
        term = sample['done']

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(actions).to(self.q_eval.device)
        term = T.tensor(term).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)

        idxs = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        # V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1,
                                            actions.unsqueeze(-1)).squeeze(-1) #[idxs, actions]

        q_next = T.add(V_s_, (A_s_ - A_s.mean(dim=1, keepdim=True)))

        q_target = rewards + self.gamma *T.max(q_next, dim=1)[0].detach() #q_next[idxs, max_actions]
        q_target[term] = 0.0
        # q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        # max_actions = T.argmax(q_eval, dim=1)

        # apply mask for terminates networks

        priorities = get_priorities(q_pred, q_target)
        self.memory.update_priorities(sample['indexes'], priorities)

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=0.5)

        self.q_eval.optimiser.step()
        self.learn_step_counter += 1
        self.dec_epsilon()

        return loss


    def test(self):
        with T.no_grad():
            self.replace_target_network()

            # sample memory
            state, actions, reward, state_, term = \
                    self.memory.sample_buffer()

            states = T.tensor(state).to(self.q_eval.device)
            actions = T.tensor(actions).to(self.q_eval.device)
            term = T.tensor(term).to(self.q_eval.device)
            rewards = T.tensor(reward).to(self.q_eval.device)
            states_ = T.tensor(state_).to(self.q_eval.device)

            idxs = np.arange(self.batch_size)

            V_s, A_s = self.q_eval.forward(states)
            V_s_, A_s_ = self.q_next.forward(states_)
            V_s_eval, A_s_eval = self.q_eval.forward(states_)

            curr_q = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[idxs, actions]

            # future q
            next_q = T.add(V_s_,(A_s_ - A_s.mean(dim=1, keepdim=True)))

            next_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

            max_actions = T.argmax(next_eval, dim=1)

            # apply mask for terminates networks
            next_eval[term] = -10.0

            q_target = rewards + self.gamma * next_q[idxs, max_actions]

            loss = self.q_eval.loss(q_target, curr_q).to(self.q_eval.device)
        return loss