"""

Notes
-----


"""

from lib.read_maze import load_maze

from ConvDDQN.classes.agent import Agent
from ConvDDQN.classes.partially_environment import Environment
from ConvDDQN.classes.canvas import Canvas

from ConvDDQN.classes.plotter import Plotter

import torchvision
import torchvision.transforms as transforms

def run(canv_chck=True, chckpt=False, train_chck=True, lr=0.01, epsilon=0.9,
        gamma=0.9, episodes=100, netname='default.pt', epsilon_min=0.01, ep_dec = 1e-4, batch_size=128, beta_inc=0.01):

    name = 'probe'
    EP = 'Random' # Random / Priority

    img_size = 37
    multi_frame = False

    if multi_frame == True:
        channels = 4
    else:
        channels = 3

    # Default (Fixed) Parameters
    epsilon_min = epsilon_min
    epsilon_dec = ep_dec
    input_dims = [channels, img_size, img_size]
    output_dims = 5

    replace_testnet = 2
    memsize = 100000 # https://arxiv.org/abs/1712.01275
    batch_size = batch_size

    env = Environment(img_size=img_size, multi_frame=multi_frame)

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=input_dims, n_actions=output_dims, mem_size=memsize, batch_size=batch_size,
                  eps_min=epsilon_min, eps_dec=epsilon_dec,
                  replace=replace_testnet, name=netname, multi_frame=multi_frame, memtype=EP)

    if canv_chck:
        canv = Canvas()
    else:
        maze = load_maze()

    if chckpt:
        agent.load_models()



    if train_chck:

        plt = Plotter(name)

        print('...start training...')
        for i in range(episodes):
            done = False
            observation = env.reset

            if canv_chck:
                canv.set_visible(env.loc.copy(), env.actor_pos, [])

            hs = None
            score = 0
            reward = 0

            while not done:
                action, acts, hs, rand = agent.greedy_epsilon(observation, hs)


                if canv_chck:
                    canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts.data.cpu().numpy(), action,
                              score, reward, env.step_cntr, i, env.wall_cntr, env.visit_cntr, env.obs2D,
                              agent.epsilon, agent.lr,
                              gamma=gamma,
                              score_split=env.get_split_score(), step_split=env.get_split_step(), rand=rand,
                              alpha=agent.alpha, beta=agent.memory.beta, EP=EP, memsize=memsize, batchsize=batch_size)


                observation_, reward, done, info = env.step(action, score)
                path = len(env.actorpath)

                score += reward
                agent.store_transition(observation, observation_, reward, action, int(done))

                loss = agent.learn()
                observation = observation_

                agent.step_params(beta_inc, env.step_cntr, i)

            if i % replace_testnet == 0:
                agent.replace_target_network()
            plt.data_in(score, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr,
                        visit_cntr=env.visit_cntr, path_cntr=path)

            print(f'Ep {i}, {loss} score {score}, epsilon {agent.epsilon}, beta {agent.memory.beta}')
            print(f'    Path Len {path} : Stayed {env.stay_cntr} : Walls {env.wall_cntr}')
            # Save NN every 10 its
            if i > 1 and i % 1 == 0:
                agent.save_models()
                plt.live_plot()


        # Need canvas, prior network and testing checked off
        print('...starting testing...')