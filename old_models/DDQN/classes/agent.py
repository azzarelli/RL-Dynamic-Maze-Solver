import numpy as np
import torch as T
from old_models.DDQN.classes.replaybuffer import PrioritizedBuffer, ReplayBuffer # Simple Replay Buffer
#from classes.replaybuffer_ import ReplayBuffer

from old_models.DDQN.classes.ddqn import DDQN


class Agent():
    def __init__(self,  n_actions, input_dims, gamma:float=0.9,  lr:float=0.01,
                 epsilon: float = 1.0, eps_min:float=0.01, eps_dec:float=5e-7,
                 mem_size:int=10000, batch_size:int=64, alpha:float=0.7, beta:float=0.4,
                 replace:int=1000,
                 save_dir:str='networkdata/', name:str='maze-test-1.pt', net_type='DDQN',
                 multi_frame: bool = True, memtype: str = 'Priority'
                 ):
        # Network parameters
        self.learn_step_counter = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_min
        self.eps_dec = eps_dec

        # PER parameters
        self.alpha = alpha
        self.beta = beta

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size

        self.replace_target_thresh = replace
        self.save_dir = save_dir

        self.action_space = [i for i in range(self.n_actions)]

        self.replay_experience = memtype
        self.multi_frame = multi_frame
        if self.replay_experience == 'Priority':
            self.memory = PrioritizedBuffer(mem_size, self.batch_size, self.alpha, self.beta)
        elif self.replay_experience == 'Random':
            # input_shape, max_size, batch_size, beta):
            self.memory = ReplayBuffer(self.input_dims, mem_size, self.batch_size, self.beta, multi_frame=self.multi_frame)


        if net_type == 'DDQN':
            self.q_eval = DDQN(self.lr, self.n_actions, input_dim=input_dims,
                               name=name, save_dir=self.save_dir)
            self.q_next = DDQN(self.lr, self.n_actions, input_dim=input_dims,
                               name=name+'.next', save_dir=self.save_dir)
            self.q_next.eval()


    def greedy_epsilon(self, observation):
        with T.no_grad():
            actions = T.Tensor([])
            # if we randomly choose max expected reward action
            if np.random.random() > self.epsilon:
                #state = (observation).to(self.q_eval.device)
                state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_eval.device)
                # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
                Q = self.q_eval.forward(state)
                action = np.argmax(Q.cpu().detach().numpy())
                actions = Q
            # otherwise random action
            else:
                action = np.random.choice(self.action_space)
            return action, actions

    def store_transition(self, state, state_, reward, action, done):
        self.memory.add(state, state_, reward, action, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_thresh == 0:
            print('Update target network...')
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_params(self,b):
        self.dec_epsilon()
        self.inc_beta(b)

    def dec_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def inc_beta(self, b):
        self.beta = self.beta + b if self.beta < 1 else 1

    def save_models(self):
        self.q_eval.save_()
        self.q_next.save_()

    def load_models(self):
        self.q_eval.load_save()
        self.q_next.load_save()

    def compute_loss(self):
        # sample memory (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights
        state, actions, reward, state_, term, weights, batch_idxs = self.memory.sample()

        #sample = self.memory.sample()
        # state = sample['state']
        # state_ = sample['state_']
        # actions = sample['action']
        # reward = sample['reward']
        # term = sample['done']

        states = T.FloatTensor(state).to(self.q_eval.device)
        actions = T.LongTensor(actions).type(T.int64) .to(self.q_eval.device)
        term = T.BoolTensor(term).to(self.q_eval.device)
        rewards = T.FloatTensor(reward).to(self.q_eval.device)
        states_ = T.FloatTensor(state_).to(self.q_eval.device)

        weights = T.FloatTensor(weights).to(self.q_eval.device)

        Q_pred = self.q_eval.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_next = self.q_next.forward(states_)

        q_pred = Q_pred
        q_next = Q_next

        q_target = rewards.squeeze(1) + self.gamma * T.max(q_next, dim=1)[0] #.detach() #q_next[idxs, max_actions]

        q_target[term] = 0.0

        # Loss
        # loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        #w = T.unsqueeze(weights, 1).to(self.q_eval.device)
        # loss = (loss * w).mean()

        td_errors = T.pow(q_pred - q_target, 2) * weights

        return td_errors, batch_idxs

    def learn(self):
        # Wait for memory to fill up before learning from empty set
        if not self.memory.is_sufficient():
            return

        #time.sleep(5) # delay for time animation
        loss, idxs = self.compute_loss()
        lossmean = loss.mean()

        self.q_eval.optimiser.zero_grad()
        lossmean.backward()

        T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=0.5)

        self.q_eval.optimiser.step()
        self.learn_step_counter += 1

        # update priorities
        for idx, td_error in zip(idxs, loss.cpu().detach().numpy()):
            self.memory.update_priorities(idx, td_error)

        return lossmean