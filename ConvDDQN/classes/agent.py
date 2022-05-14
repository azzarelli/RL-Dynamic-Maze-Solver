import numpy as np
import torch
import torch as T
import sys

from ConvDDQN.classes.replaybuffer import PrioritizedBuffer, RandomBuffer # Simple Replay Buffer

from ConvDDQN.classes.convddqn_lstm import ConvDDQN as ConvDDQNLSTM
from ConvDDQN.classes.convdqn_lstm import ConvDQN as ConvDQNLSTM

import torch.nn as nn
import torch

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='networkdata/', name='maze-test-1.pt', multi_frame:bool=True, memtype:str='Random',
                 alpha=0.5, beta=0.5, loss_type:str='MSE', net_type='DDQN'):
        '''Define Network Parameters'''
        self.learn_step_counter = 0 # used to update target network
        self.gamma = gamma
        self.lr = lr # learning rate
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)] # define action space
        self.input_dims = input_dims
        self.batch_size = batch_size

        '''Epsilon Greedy Linear Annealing Parameters (Note not all the parameters may be relevant to the greedy
         algorithm which exists but can be used to change exploration strategy)'''
        self.epsilon = epsilon # initialise epsilon start value
        self.eps_start = epsilon # if necessary initialise the start of epsilon at the beginning of memory
        self.eps_min = eps_min # if necessary initialise
        self.eps_dec = eps_dec # rate of epsilon decrease for linear annealing

        '''Saving & Loading Network Parameters'''
        self.replace_target_thresh = replace
        self.save_dir = save_dir

        '''Choice of Experience Replay'''
        self.replay_experience = memtype
        self.multi_frame = multi_frame # choice of using multiple frames for for input (not advised)
        if self.replay_experience == 'Priority': # PER
            self.alpha = alpha # prioritisation parameter for TDError calcuation from forward passs
            self.beta = beta # prioritisation of
            self.memory = PrioritizedBuffer(mem_size, self.batch_size, self.alpha, self.beta)
        elif self.replay_experience == 'Random': # Random ER
            self.memory = RandomBuffer(self.input_dims, mem_size, self.batch_size, multi_frame=self.multi_frame)
        else:
            print('Error: Incorrectly defined memory type.')
            sys.exit()

        '''Choce of DQN variants'''
        loss_type = loss_type # Define loss type which we use for training
        network_type = net_type
        if network_type == 'DQN': # Vanilla DQN with LSTM
            self.q_eval = ConvDQNLSTM(self.lr, self.n_actions, input_dim=input_dims,
                                   name=name, save_dir=self.save_dir, loss_type=loss_type)
            self.q_next = ConvDQNLSTM(self.lr, self.n_actions, input_dim=input_dims,
                                   name=name + 'next', save_dir=self.save_dir, loss_type=loss_type)
        elif network_type == 'DDQN': # Duelling DQN with LSTM
            self.q_eval = ConvDDQNLSTM(self.lr, self.n_actions, input_dim=input_dims,
                               name=name, save_dir=self.save_dir, loss_type=loss_type)
            self.q_next = ConvDDQNLSTM(self.lr, self.n_actions, input_dim=input_dims,
                               name=name+'next', save_dir=self.save_dir, loss_type=loss_type)
        else:
            print('Error: Incorrectly defined network type.')
            sys.exit()

        self.q_next.eval()  # as we aren't training q_next but updating through loading network states, we can remove it from computational graph
        self.freeze_network() # as a redundancy we freeze parameters in network to prevent gradient flow later

    def freeze_network(self):
        """To prevent gradients from flowing through/from target network we want to (re-)freeze the layers of
        the target network
        """
        for parameters in self.q_next.parameters():
            parameters.requires_grad_(requires_grad=False)

    def greedy_epsilon(self, observation, hs):
        """Exploration Strategy defined by epsilon greedy algorithm
        (random choice taken when random variable is greater than epsilon)
        """
        with T.no_grad(): # Detach action from computational graph as we aren't training algorithm at this step
            if np.random.random() > self.epsilon:
                state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_eval.device)
                Q, hs = self.q_eval(state, hs)
                actions = Q
                action = np.argmax(Q.cpu().detach().numpy()) # fetch max Q vale for action
                rand=False # output flag denoting if the action taken was random or not

                return action, actions, hs, rand

            else:
                action = np.random.choice(self.action_space)
                rand=True

                return action, torch.Tensor([]), hs, rand

    def store_transition(self, state, state_, reward, action, done):
        """Store experience in memory"""
        self.memory.add(state, state_, reward, action, done)

    def replace_target_network(self):
        """Replace the Target network with newly updated network"""
        print('Update target network...')
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()
        self.freeze_network() # refreeze loaded network

    def step_params(self,b, step, episode, avg=0):
        """Re-setting/reconfiguring active parameters (can be called after each epoch/step dependant on need)"""
        self.dec_epsilon(step, episode, avg)
        self.inc_beta(b)

    def dec_epsilon(self, step, episode, avg):
        """Explicit method of anealing epsilon for optimal exploration"""
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > 0.05 else 0.05
        c = (-0.012*avg )/100+0.4

        # Bound the epsilon
        if self.epsilon > 0.8: self.epsilon = 0.8
        elif self.epsilon < 0.2: self.epsilon = 0.2

    def inc_beta(self, b):
        """As we generate more experience we want to anneal beta to reduce likelihood of fetching a low priority (& old)
        experience
        """
        if self.replay_experience == 'Priority':
            self.memory.beta = self.memory.beta + b if self.memory.beta < 1 else 1

    def save_models(self):
        """Save both models incase of unexpected crash (reset GPU kernel if this is the case)"""
        self.q_eval.save_()
        self.q_next.save_()

    def load_models(self):
        """Load models if they exist"""
        self.q_eval.load_save()
        self.q_next.load_save()

    def compute_loss(self):
        """Computing the loss of learn-step

        Notes
        -----
        We fetch experience (and experience weights) from memory and pass values through the training network (`q_eval`,
        not to be confused with the target network `q_next`). LSTM flavours will need a batch of hidden state values
        (`hs`), thus we initialise empty state for all experiences in batch
        """
        '''Sample experiences from memory'''
        if self.replay_experience == 'Priority':
            state, actions, reward, state_, term, weights, batch_idxs = self.memory.sample()
            weights = T.FloatTensor(weights).to(self.q_eval.device)
        elif self.replay_experience == 'Random':
            state, actions, reward, state_, term = self.memory.sample()

        states = T.FloatTensor(state).to(self.q_eval.device)
        actions = T.LongTensor(actions).type(T.int64) .to(self.q_eval.device)
        term = T.BoolTensor(term).to(self.q_eval.device)
        rewards = T.FloatTensor(reward).to(self.q_eval.device)
        states_ = T.FloatTensor(state_).to(self.q_eval.device)

        '''Forward experiences (+ hidden cells) through training network'''
        hs = (T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_eval.device), T.autograd.Variable(T.zeros(1, 512).float()).to(self.q_eval.device))
        q_pred, hs_ = self.q_eval(states, hs)
        Q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1) # fetch q-values for actions defined in experience

        '''Fetch Target values '''
        Q_next, hs = self.q_next(states_, hs)

        q_pred = Q_pred
        q_next = Q_next

        '''Determine the target Q values throughout episode by defining future rewards + reward of current action'''
        if self.replay_experience == 'Priority':
            q_target = rewards.squeeze(1) + self.gamma * T.max(q_next, dim=1)[0]
        elif self.replay_experience == 'Random':
            q_target = rewards.squeeze(0) + self.gamma * T.max(q_next, dim=1)[0]

        '''Mask Target values if episode terminated at this point'''
        q_target[term] = 0.0
        q_target.detach_()

        '''Loss calculation'''
        if self.replay_experience == 'Priority':
            td_errors = self.q_eval.loss(q_pred, q_target) * weights.detach()
        elif self.replay_experience == 'Random':
            td_errors = self.q_eval.loss(q_pred, q_target)
            batch_idxs = []

        return td_errors, batch_idxs

    def learn(self):
        """Handle the learning capability

        Notes
        -----
        We first check if learning should happend (dependant on if memory is sufficient). Thereafter we run the compute
        loss method and back-propagrate + step the optimiser. In addition, when memory is PER we update the priorities
        of the step.
        """
        # time.sleep(5) # delay for canvas-visual analysis

        if not self.memory.is_sufficient():
            return

        '''Set Optimiser and compute loss'''
        self.q_eval.optimiser.zero_grad()
        loss, idxs = self.compute_loss()

        '''PER, requires mean calculation of loss as TDErrors are passed through as a batch for PER, this is done so
        we can immediately pass TDError through to PER'''
        if self.replay_experience == 'Priority':
            lossmean = loss.mean()
            lossmean.backward()

            with torch.no_grad(): # don't mess with gradient flow
                for idx, td_error in zip(idxs, loss.cpu().detach().numpy()): # pass error through
                        self.memory.update_priorities(idx, td_error.item()) # updat experience priorities

        elif self.replay_experience == 'Random':
            lossmean = loss
            loss.backward()

        self.q_eval.optimiser.step()
        self.learn_step_counter += 1

        return lossmean.cpu() # return loss to `run.py`