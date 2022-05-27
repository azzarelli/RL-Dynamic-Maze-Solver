"""Main execution function
        - tunable paramerters included to control function of the algorithm

"""
from ConvDDQN.run import run

if __name__ == '__main__':
    iter = [0] # We use for loops for hyper parameter tuning
    for j in range(1):
        for i in iter:
            NAME = 'eps0.4_0' # Name of the network
            '''
                canv_chck: int, frequency of showing an epsidoe using pygame canvas (every n episodes) where n > -1
                train_chck: bool, True for training otherwise False
                chckpt: bool, True/False for initialising our network with an existing network named `NAME` (cached networks found in `network data`)
                episodes, lr, gamma, batch_size: network hyper parameters
                epsilon: float, denotes the starting value of epsilon
                ep_dec: fload, denote the initial value for increading epsilon between steps (this is varied in the agent handler)
                 
            '''
            run(canv_chck=0, train_chck=True, chckpt=False, netname=NAME,
                episodes=1000, lr=0.001, gamma=0.99, batch_size=64,
                epsilon=0.4, ep_dec=0.05)

