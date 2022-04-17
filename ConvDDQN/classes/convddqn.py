import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


test_list = []

class ConvDDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, save_dir):
        super(ConvDDQN, self).__init__()
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, name)

        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.fc_input_dim = self.feature_size()

        self.value_stream = nn.Sequential(
            nn.Linear(33856, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(33856, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # self.loss = nn.HuberLoss()
        #self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... {name} Network training on {self.device} ...')
        self.to(self.device)

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        V = self.value_stream(features)
        A = self.advantage_stream(features)
        Q = V + (A - A.mean())
        return Q

    def feature_size(self):
        return self.conv(T.autograd.Variable(T.zeros(1, *self.input_dim))).view(1, -1).size(1)


    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))