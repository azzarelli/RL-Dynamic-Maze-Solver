import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, save_dir):
        super(DQN, self).__init__()
        self.save_dir = save_dir
        self.save_file = os.path.join(self.save_dir, name)


        self.feature_stream = nn.Sequential( # Convolutional layer
            nn.Linear(*input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            )
        self.sm = nn.Softmax(dim=1)

        self.optimiser = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.L1Loss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(f'... {name} Network training on {self.device} ...')
        self.to(self.device)

    def forward(self, state):
        action = self.feature_stream(state)
        return action

    def save_(self):
        print('Saving network ...')
        T.save(self.state_dict(), self.save_file)

    def load_save(self): # file
        print('Load saves ...')
        self.load_state_dict(T.load(self.save_file))