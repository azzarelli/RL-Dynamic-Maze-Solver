import numpy as np
import torch
import torch as T
import sys

from ConvDDQN.classes.replaybuffer import PrioritizedBuffer, RandomBuffer # Simple Replay Buffer
from torchvision.utils import save_image

from ConvDDQN.classes.convddqn_lstm import ConvDDQN as ConvDDQNLSTM
from ConvDDQN.classes.convdqn_lstm import ConvDQN as ConvDQNLSTM

import torch.nn as nn
import torch

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, save_dir='networkdata/', name='maze-test-1.pt', multi_frame:bool=True, memtype:str='Random',
                 alpha=0.6, beta=0.4, loss_type:str='MSE', net_type='DDQN'):
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

        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.q_eval.optimiser, step_size=50, gamma=.9)
        self.q_next.cuda()
        self.q_eval.cuda()

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
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                state = T.FloatTensor(observation).float().unsqueeze(0).to(self.q_eval.device)
                Q, hs = self.q_eval(state, hs)
                actions = Q
                action = np.argmax(Q.cpu().data.numpy()) # fetch max Q vale for action
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
        self.q_next.load_state_dict(self.q_eval.state_dict())
        self.q_next.eval()
        self.freeze_network() # refreeze loaded network

    def replace_target_network_soft(self):
        """Replace the Target network with newly updated network (soft update)
            targetnet = tau*evalnet + (1-tau) *targetnet
        """
        with torch.no_grad():
            tau = 0.001
            for t_param, ev_param in zip(self.q_next.parameters(), self.q_eval.parameters()):
                t_param.data.copy_(tau*t_param + (1-tau)*ev_param)
                assert t_param.requires_grad == False
            self.q_next.eval()
            self.freeze_network() # refreeze loaded network

    def step_params(self, beta, step, episode, avg=0):
        """Re-setting/reconfiguring active parameters (can be called after each epoch/step dependant on need)"""
        self.dec_epsilon(step, episode, avg)
        self.inc_beta(beta)

    def dec_epsilon(self, step, episode, avg):
        """Explicit method of anealing epsilon for optimal exploration"""

        if step == 1:
            self.eps_start = self.eps_start - 0.01 if self.eps_start > 0.05 else 0.05
            self.epsilon = self.eps_start
        else:
            self.epsilon = self.epsilon + self.eps_dec if self.epsilon < 0.8 else 0.8

        # Bound the epsilon
        # if self.epsilon > 0.8: self.epsilon = 0.8
        # elif self.epsilon < 0.1: self.epsilon = 0.1

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
        actions = T.LongTensor(actions).type(T.int64).to(self.q_eval.device)
        term = T.BoolTensor(term).to(self.q_eval.device)
        rewards = T.FloatTensor(reward).to(self.q_eval.device)
        states_ = T.FloatTensor(state_).to(self.q_eval.device)

        # Visualise current and next state batch (batch printed as entire image)
        # save_image(states, 'now.png')
        # save_image(states_, 'next.png')

        '''Forward experiences (+ hidden cells) through training network'''
        hs = (T.autograd.Variable(T.zeros(1, 64).float()).to(self.q_eval.device), T.autograd.Variable(T.zeros(1, 64).float()).to(self.q_eval.device))
        Q_pred, hs_ = self.q_eval(states, hs)

        Q_pred = Q_pred.gather(1, actions.unsqueeze(1)).squeeze(1) # fetch q-values for actions defined in experience


        with torch.no_grad():
            '''Fetch Target values '''
            q_next, hs = self.q_next(states_, hs_)

            '''Determine the target Q values throughout episode by defining future rewards + reward of current action'''
            if self.replay_experience == 'Priority':
                Q_target = (rewards.squeeze(1) + self.gamma * T.max(q_next, dim=1)[0]).to(self.q_next.device).detach()
            elif self.replay_experience == 'Random':
                Q_target = (rewards.squeeze(0) + self.gamma * T.max(q_next, dim=1)[0]).to(self.q_next.device).detach()

            '''Mask Target values if episode terminated at this point'''
            Q_target[term] = 0.0
        '''Loss calculation'''
        if self.replay_experience == 'Priority':
            loss = self.q_eval.loss(Q_target, Q_pred).to(self.q_eval.device) # loss function
            tderror = T.absolute(Q_target.detach() - Q_pred.detach()).cpu().numpy() # td error is used for updating priorities
            loss = loss * weights.detach()
        elif self.replay_experience == 'Random':
            loss = self.q_eval.loss(Q_target, Q_pred).to(self.q_eval.device)
            tderror = 0
            batch_idxs = []

        return loss, tderror, batch_idxs

    def learn(self):
        """Handle the learning capability & Updating the network

        Notes
        -----
        We first check if learning should happend (dependant on if memory is sufficient). Thereafter we run the compute
        loss method and back-propagrate + step the optimiser. In addition, when memory is PER we update the priorities
        of the step.
        """
        # time.sleep(5) # delay for canvas-visual analysis

        if not self.memory.is_sufficient():
            return

        '''Set Optimiser and Compute Loss'''
        loss, td_error, idxs = self.compute_loss()
        self.q_eval.optimiser.zero_grad()

        '''PER, requires mean calculation of loss as TDErrors are passed through as a batch for PER, this is done so
        we can immediately pass TDError through to PER'''
        if self.replay_experience == 'Priority':
            lossmean = loss.mean()
            lossmean.backward()
            for idx, td in zip(idxs, td_error): # pass error through
                    self.memory.update_priorities(idx, td+0.0001) # update experience priorities + non-zeroing small error term


        elif self.replay_experience == 'Random':
            lossmean = loss
            loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.q_eval.parameters(), 1.)
        self.q_eval.optimiser.step()
        #self.scheduler.step()
        self.learn_step_counter += 1

        self.replace_target_network_soft()

        return lossmean.item() # return loss to `run.py`