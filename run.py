"""

Notes
-----


"""

from lib.read_maze import load_maze

from classes.agent import Agent
from classes.environment import Environment
from classes.canvas import Canvas

from classes.plotter import Plotter


def run(canv_chck=True, chckpt=False, train_chck=True, lr=0.01, epsilon=0.9,
        gamma=0.9, episodes=100, netname='default.pt', epsilon_min=0.01, batch_size=128):

    name = 'l1_512-l2_1024'

    # Default (Fixed) Parameters
    epsilon_min = epsilon_min
    epsilon_dec = 1e-4
    input_dims = [29]
    output_dims = 5

    replace_testnet = 20
    memsize = 10000 # https://arxiv.org/abs/1712.01275
    batch_size = batch_size

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=input_dims, n_actions=output_dims, mem_size=memsize, eps_min=epsilon_min,
                  batch_size=batch_size, eps_dec=epsilon_dec, replace=replace_testnet, name=netname)

    if canv_chck:
        canv = Canvas()
    else:
        maze = load_maze()

    if chckpt:
        agent.load_models()

    env = Environment()

    if train_chck:

        plt = Plotter(name)

        print('...start training...')
        for i in range(episodes):
            done = False
            observation = env.reset

            if canv_chck:
                canv.set_visible(env.get_local_matrix.copy(), env.actor_pos, [])

            score = 0
            while not done:
                action, acts = agent.greedy_epsilon(observation)
                observation_, reward, done, info = env.step(action, score)

                path = len(env.actorpath)

                score += reward
                agent.store_transition(observation, observation_, reward, action,
                                       int(done))

                loss = agent.learn()
                observation = observation_

                if canv_chck:
                    canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts.data.cpu().numpy(), action)

            plt.data_in(score, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr,
                        visit_cntr=env.visit_cntr, path_cntr=path)
            print(f'Ep {i}, {loss} score {score}, avg {plt.scores_avg[-1]}, epsilon {agent.epsilon}')
            print(f'    Path Len {path} : Stayed {env.stay_cntr} : Walls {env.wall_cntr}')
            # Save NN every 10 its
            if i > 20 and i % 20 == 0:
                agent.save_models()
                plt.live_plot()


    elif not train_chck and canv_chck and chckpt:
        agent = Agent(gamma=gamma, epsilon=0., lr=lr,
                      input_dims=input_dims, n_actions=output_dims, mem_size=memsize,
                      batch_size=batch_size, eps_dec=epsilon_dec, replace=replace_testnet, name=netname)

        # Need canvas, prior network and testing checked off
        print('...starting testing...')
        for i in range(episodes):
            done = False
            observation = env.reset

            canv.set_visible(env.get_local_matrix.copy(), env.actor_pos, [])

            score = 0
            while not done:
                action, acts = agent.greedy_epsilon(observation)
                observation_, reward, done, info = env.step(action, score)

                score += reward
                agent.store_transition(observation, observation_, reward, action,
                                       int(done))

                canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts)

            print(f'Ep {i}, score {score}, lr {lr}')