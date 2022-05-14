import numpy as np
import torch as T
import time
from ConvDDQN.classes.replaybuffer import PrioritizedBuffer, RandomBuffer # Simple Replay Buffer
#from classes.replaybuffer_ import ReplayBuffer

from ConvDDDQN.classes.convddqn import ConvDDQN
from torchvision.transforms import  transforms


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

        self.q_model1 = ConvDDQN(self.lr, self.n_actions, input_dim=input_dims,
                           name=name, save_dir=self.save_dir)
        self.q_model2 = ConvDDQN(self.lr, self.n_actions, input_dim=input_dims,
                           name=name+'next', save_dir=self.save_dir)
        # self.q_next.eval()


    def greedy_epsilon(self, observation, hs):
        with T.no_grad():
            actions = T.Tensor([])
            # if we randomly choose max expected reward action
            #state = (observation).to(self.q_eval.device)
            state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_model1.device)
            # state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            Q, hs = self.q_model1.forward(state, hs)
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

    # def replace_target_network(self):
    #     if self.learn_step_counter % self.replace_target_thresh == 0:
    #         print('Update target network...')
    #         self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_params(self,b, step, episode):
        self.dec_epsilon(step, episode)
        self.inc_beta(b)

    def dec_epsilon(self, step, episode):
        a = 0.9  # max factor
        x = step + 1  # step in current approach
        f = episode + 1  # episode

        if episode > 6000:
            b = 1. / float(f)
            self.epsilon = a * np.arctan((b * x)) / (np.pi / 2)
        elif episode > 4000:
            b = 2. / float(f)
            self.epsilon = a * np.arctan((b * x)) / (np.pi / 2)
        elif episode > 2000:
            b = 4. / float(f)
            self.epsilon = a * np.arctan((b * x)) / (np.pi / 2)
        else:
            b = 4. / float(f)
            self.epsilon = a * np.arctan((b * x)) / (np.pi / 2)

    def inc_beta(self, b):
        self.memory.beta = self.memory.beta + b if self.memory.beta < 1 else 1

    def save_models(self):
        self.q_model1.save_()
        self.q_model2.save_()

    def load_models(self):
        self.q_model1.load_save()
        self.q_model2.load_save()

    def compute_loss(self):
        if self.replay_experience == 'Priority':
            state, actions, reward, state_, term, weights, batch_idxs = self.memory.sample()
        elif self.replay_experience == 'Random':
            state, actions, reward, state_, term = self.memory.sample()

        states = T.FloatTensor(state).to(self.q_model1.device)
        actions = T.LongTensor(actions).type(T.int64) .to(self.q_model1.device)
        term = T.BoolTensor(term).to(self.q_model1.device)
        rewards = T.FloatTensor(reward).to(self.q_model1.device)
        states_ = T.FloatTensor(state_).to(self.q_model1.device)

        if self.replay_experience == 'Priority':
            weights = T.FloatTensor(weights).to(self.q_model1.device)

        hs = (T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_model1.device), T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_model1.device))
        q_model1, hs_1 = self.q_model1.forward(states, hs)
        q_model2, hs_2 = self.q_model1.forward(states, hs)
        Q_model1 = q_model1.gather(1, actions.unsqueeze(1)).squeeze(1)
        Q_model2 = q_model2.gather(1, actions.unsqueeze(1)).squeeze(1)

        q_next1, hs_ = self.q_model1.forward(states_, hs)
        q_next2, hs_ = self.q_model2.forward(states_, hs)

        Q_next = T.min(
            T.max(q_next1, dim=1)[0],
            T.max(q_next2, dim=1)[0]
        )
        Q_next = Q_next.view(Q_next.size(0), 1)

        if self.replay_experience == 'Priority':
            Q_target = rewards.squeeze(1) + self.gamma * Q_next
        elif self.replay_experience == 'Random':
            Q_target = rewards.squeeze(0) + self.gamma * Q_next

        Q_target[term] = 0.0

        # perhaps use q_trarget.detach()
        if self.replay_experience == 'Priority':
            td_errors1 = self.q_model1.loss(Q_model1,Q_target.detach()).to(self.q_model1.device) * weights
            td_errors2 = self.q_model2.loss(Q_model2, Q_target.detach()).to(self.q_model2.device) * weights
        elif self.replay_experience == 'Random':
            td_errors1 = self.q_model1.loss(Q_model1, Q_target.detach()).to(self.q_model1.device)
            td_errors2 = self.q_model2.loss(Q_model2, Q_target.detach()).to(self.q_model2.device)
            batch_idxs = []
        return td_errors1, td_errors2, batch_idxs

    def learn(self):
        # Wait for memory to fill up before learning from empty set
        if not self.memory.is_sufficient():
            return

        # with T.no_grad():

        # time.sleep(5) # delay for time animation
        self.q_model1.train()
        self.q_model2.train()

        self.q_model1.optimiser.zero_grad()
        self.q_model2.optimiser.zero_grad()

        loss1, loss2, idxs = self.compute_loss()

        if self.replay_experience == 'Priority':
            lossmean1 = loss1.mean()
            lossmean2 = loss2.mean()
            lossmean1.backward()
            lossmean2.backward()

            # update priorities
            for idx, td_error, td_error2 in zip(idxs, loss1.cpu().detach().numpy(), loss2.cpu().detach().numpy()):
                self.memory.update_priorities(idx, td_error)

        elif self.replay_experience == 'Random':
            lossmean1 = loss1
            loss1.backward()
            loss2.backward()


        # T.nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=0.5)

        self.q_model1.optimiser.step()
        self.q_model2.optimiser.step()
        self.learn_step_counter += 1
        # self.dec_epsilon()
        # self.inc_beta(0.01)

        return lossmean1