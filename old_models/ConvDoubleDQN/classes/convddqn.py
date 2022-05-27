import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

test_list = []

class ConvDDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, save_dir):
        super(ConvDDQN, self).__init__()
        self.save_dir = save_dir
        self.save_file = self.save_dir + name+'.pt' #os.path.join(self.save_dir, name)

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
        self.fc_input_dim = self._get_conv_out(input_dim)
        self.feature_stream = nn.Sequential(
            # nn.Linear(64, 256),
            # nn.ReLU(),
            nn.Linear(self.n_hidden, 1)
        )


        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        #self.loss = nn.SmoothL1Loss()
        # self.loss = nn.HuberLoss()
        #self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... {name} Network training on {self.device} ...')
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(T.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, state, hs=None):
        features = self.conv(state)
        features = features.view(features.size(0), -1)

        features, hs_ = self.lstm_layer(features, hs)
        features = features.reshape(-1, self.n_hidden)

        Q = self.feature_stream(features)
        return Q, hs_

    def feature_size(self):
        return self.conv(T.autograd.Variable(T.zeros(1, *self.input_dim))).view(1, -1).size(1)


    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))