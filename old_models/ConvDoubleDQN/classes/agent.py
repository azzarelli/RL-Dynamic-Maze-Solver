import numpy as np
import torch as T
from old_models.ConvDoubleDQN.classes.replaybuffer import PrioritizedBuffer, RandomBuffer # Simple Replay Buffer

from old_models.ConvDoubleDQN.classes.convddqn import ConvDDQN


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='networkdata/', name='maze-test-1.pt', multi_frame:bool=True, memtype:str='Random',
                 alpha=0.6, beta=0.5):
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

        self.replay_experience = memtype

        self.action_space = [i for i in range(self.n_actions)]

        self.multi_frame = multi_frame
        if self.replay_experience == 'Priority':
            self.memory = PrioritizedBuffer(mem_size, self.batch_size, self.alpha, self.beta)
        elif self.replay_experience == 'Random':
            # input_shape, max_size, batch_size, beta):
            self.memory = RandomBuffer(self.input_dims, mem_size, self.batch_size, self.beta, multi_frame=self.multi_frame)

        self.q_eval = ConvDDQN(self.lr, self.n_actions, input_dim=input_dims,
                           name=name, save_dir=self.save_dir)
        self.q_next = ConvDDQN(self.lr, self.n_actions, input_dim=input_dims,
                           name=name+'next', save_dir=self.save_dir)
        # self.q_next.eval()


    def greedy_epsilon(self, observation, hs):
        with T.no_grad():
            actions = T.Tensor([])
            # if we randomly choose max expected reward action
            #state = (observation).to(self.q_eval.device)
            state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_eval.device)
            # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            Q, hs = self.q_eval.forward(state, hs)
            actions = Q
            # otherwise random action
            if np.random.random() > self.epsilon:
                action = np.argmax(Q.cpu().detach().numpy())
                # print(actions, action)
                rand=False
            else:
                action = np.random.choice(self.action_space)
                rand=True
            return action, actions, hs, rand

    def store_transition(self, state, state_, reward, action, done):
        self.memory.add(state, state_, reward, action, done)

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_thresh == 0:
            print('Update target network...')
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_params(self,b, step, episode, inc_grad):
        self.dec_epsilon(step, episode, inc_grad)
        self.inc_beta(b)

    def dec_epsilon(self, step, episode, inc_grad):
        # self.epsilon = self.epsilon - 0.01 if self.epsilon-0.01 > 0.05 else 0.05
        # self.epsilon = 0.3
        a = 0.9 # max factor
        x = step + 1  # step in current approach
        f = episode + 1  # episode
        c = 6
        b = 1/(c*float(f))
        self.epsilon = a * np.arctan((b * x)) / (np.pi / 2)

    def inc_beta(self, b):
        self.memory.beta = self.memory.beta + b if self.memory.beta < 1 else 1

    def save_models(self):
        self.q_eval.save_()
        self.q_next.save_()

    def load_models(self):
        self.q_eval.load_save()
        self.q_next.load_save()

    def compute_loss(self):
        if self.replay_experience == 'Priority':
            state, actions, reward, state_, term, weights, batch_idxs = self.memory.sample()
        elif self.replay_experience == 'Random':
            state, actions, reward, state_, term = self.memory.sample()

        states = T.FloatTensor(state).to(self.q_eval.device)
        actions = T.LongTensor(actions).type(T.int64) .to(self.q_eval.device)
        term = T.BoolTensor(term).to(self.q_eval.device)
        rewards = T.FloatTensor(reward).to(self.q_eval.device)
        states_ = T.FloatTensor(state_).to(self.q_eval.device)

        if self.replay_experience == 'Priority':
            weights = T.FloatTensor(weights).to(self.q_eval.device)

        hs = (T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_eval.device), T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_eval.device))
        q_pred, hs_ = self.q_eval.forward(states, hs)
        Q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        with T.no_grad():
            Q_next, hs = self.q_next.forward(states_, hs_)

        q_pred = Q_pred
        q_next = Q_next
        if self.replay_experience == 'Priority':
            q_target = rewards.squeeze(1) + self.gamma * T.max(q_next, dim=1)[0] #.detach() #q_next[idxs, max_actions]
        elif self.replay_experience == 'Random':
            q_target = rewards.squeeze(0) + self.gamma * T.max(q_next, dim=1)[0] #.detach() #q_next[idxs, max_actions]

        q_target[term] = 0.0

        # Loss
        # loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        #w = T.unsqueeze(weights, 1).to(self.q_eval.device)
        # loss = (loss * w).mean()

        # perhaps use q_trarget.detach()
        if self.replay_experience == 'Priority':
            td_errors = self.q_eval.loss(q_target.detach(), q_pred).to(self.q_eval.device) * weights
        elif self.replay_experience == 'Random':
            td_errors = self.q_eval.loss(q_target.detach(), q_pred).to(self.q_eval.device)
            batch_idxs = []
        return td_errors, batch_idxs

    def learn(self):
        # Wait for memory to fill up before learning from empty set
        if not self.memory.is_sufficient():
            return

        # with T.no_grad():

        # time.sleep(5) # delay for time animation
        self.q_eval.train()
        self.q_eval.optimiser.zero_grad()
        loss, idxs = self.compute_loss()

        if self.replay_experience == 'Priority':
            lossmean = loss.mean()
            lossmean.backward()

            # update priorities
            for idx, td_error in zip(idxs, loss.cpu().detach().numpy()):
                self.memory.update_priorities(idx, td_error)

        elif self.replay_experience == 'Random':
            lossmean = loss
            loss.backward()


        # T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=0.5)

        self.q_eval.optimiser.step()
        self.learn_step_counter += 1


        return lossmean