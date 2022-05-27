"""Run Training/Testing

Notes
-----
We pass through tunable parameters. Within the function there are a set of untanble parameters we can set to run different
styles of architecture, such as {DQN, DDQN} or {L1, Huber, MSE, SmoothL1} loss. We can also change the frame-size and
input image size from here.

"""

from ConvDDQN.lib.read_maze import load_maze
from ConvDDQN.classes.agent import Agent
from ConvDDQN.classes.partially_environment import Environment
from ConvDDQN.classes.canvas import Canvas
from ConvDDQN.classes.training import train, test


def run(canv_chck=True, chckpt=False, train_chck=True, lr=0.01, epsilon=0.9,
        gamma=0.9, episodes=100, netname='default.pt', epsilon_min=0.01, ep_dec = 1e-4, batch_size=128, beta_inc=0.01):
    '''Define name of network and type of experience replay (memory)'''
    name = netname # save/load file name
    net_type = 'DDQN' # {DQN, DDQN}
    loss_type = 'L1' # {MSE, L1, SmoothL1, Huber}
    EP = 'Random' # {Random, Priority}

    '''Parameters modifying input image of convolutional networks'''
    img_size = 10
    multi_frame = False # multiple frames (not suggested as our game doesn't have 'motion-blur') - now depricated as not necessary
    channels = 3 # colour image input

    '''Parameters for exploration'''
    epsilon_min = epsilon_min
    epsilon_dec = ep_dec

    '''Network Parameters'''
    replace_testnet = 10 # frequency of replacing target network with evaluation network
    input_dims = [channels, 40, 40]  # input dimensions to network
    output_dims = 5  # size of action space
    memsize = 300000 # capacity of memory
    batch_size = batch_size # batch size

    '''Initialise Handlers for Environment & Agent '''
    env = Environment(img_size=img_size) # enviornment handler
    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=input_dims, n_actions=output_dims, mem_size=memsize, batch_size=batch_size,
                  eps_min=epsilon_min, eps_dec=epsilon_dec,
                  replace=replace_testnet, name=netname, multi_frame=multi_frame, memtype=EP,
                  loss_type=loss_type, net_type=net_type) # agent handler
    if chckpt:
        agent.load_models() # load model state if we created a model checkpoint in a prior run

    '''Initialise canvas if visualising training (note this method is 3x slower than training without canvas)'''
    if canv_chck:
        canv = Canvas()
    else:
        canv=None
        maze = load_maze() # this function is initialised in Canvas, so initialise our maze here if no Canvas is called

    '''Training'''
    if train_chck:
        print('...start training...')
        train(name, episodes, gamma, memsize, batch_size, env, agent, canv, canv_chck, beta_inc, replace_testnet, EP)

    '''Testing'''
    print('...start training...')
    test(gamma, memsize, batch_size, env, agent, canv, EP)


