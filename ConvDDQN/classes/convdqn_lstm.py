"""Convolutional DQN with LSTM flavour
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ConvDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, save_dir, loss_type:str='MSE'):
        super(ConvDQN, self).__init__()
        '''Save & Load Directories'''
        self.save_dir = save_dir
        self.save_file = self.save_dir + name+'.pt'


        '''Define DQN Network'''
        self.input_dim = input_dim
        self.num_actions = n_actions
        self.n_hidden = 512

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.lstm_layer = nn.LSTM(input_size=64, hidden_size=self.n_hidden, num_layers=1, batch_first=True)
        self.feature_stream = nn.Sequential(
            # nn.Linear(64, 256),
            # nn.ReLU(),
            nn.Linear(self.n_hidden, n_actions)
        )

        '''Optimiser & Loss configuration'''
        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        if loss_type == 'MSE':
            self.loss = nn.MSELoss()
        elif loss_type == 'SmoothL1':
            self.loss = nn.SmoothL1Loss()
        elif loss_type == 'Huber':
            self.loss = nn.HuberLoss()
        elif loss_type == 'L1':
            self.loss = nn.L1Loss()

        '''CPU/GPU device configuration (use of GPU is advised)'''
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... {name} Network training on {self.device} ...')
        self.to(self.device)

    def forward(self, state, hs=None):
        features = self.conv(state) # DDQN forward pass
        features = features.view(features.size(0), -1)

        features, hs_ = self.lstm_layer(features, hs) # modify feature values for lstm input
        features = features.reshape(-1, self.n_hidden) # pass hidden state & modified feature state to lstm

        Q = self.feature_stream(features) # pass lstm features through last DQN layer
        return Q, hs_

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))