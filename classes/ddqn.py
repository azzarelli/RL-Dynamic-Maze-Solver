import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


test_list = []

class DDQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dim, save_dir):
        super(DDQN, self).__init__()
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, name)

        self.features = nn.Sequential( # Convolutional layer
            nn.Linear(*input_dim, 1024),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_actions)
        )

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # self.loss = nn.HuberLoss()
        #self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... {name} Network training on {self.device} ...')
        self.to(self.device)

    def forward(self, state):
        features = self.features(state)
        V = self.value_stream(features)
        A = self.advantage_stream(features)

        Q = T.add(V, (A - A.mean()))
        return Q

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))