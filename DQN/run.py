"""

Notes
-----


"""
import time

from lib.read_maze import load_maze

from classes.agent import Agent
from classes.environment import Environment
from classes.canvas import Canvas

from classes.plotter import Plotter

import argparse


def run(train_chck=True, chckpt=False, lr=0.01, epsilon=0.9,
        gamma=0.99, episodes=100, netname='default.pt'):
    # Default (Fixed) Parameters
    epsilon_min = 0.1
    epsilon_dec = 7e-5
    input_dims = [18]
    output_dims = 5

    replace_testnet = 50
    memsize = 1000000 # https://arxiv.org/abs/1712.01275
    batch_size = 64

    # CER buffer
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-bs', type=int, default=1000)
    parser.add_argument('-cer', type=bool, default=False)
    # if you supply it, then true

    agent = Agent(gamma=gamma, epsilon=epsilon, lr=lr,
                  input_dims=input_dims, n_actions=output_dims, mem_size=memsize, eps_min=epsilon_min,
                  batch_size=batch_size, eps_dec=epsilon_dec, replace=replace_testnet, name=netname)

    if not train_chck:
        canv = Canvas()
    else:
        maze = load_maze()

    if chckpt:
        agent.load_models()

    plt = Plotter()
    env = Environment()
    print('...starting...')
    for i in range(episodes):
        done = False
        observation = env.reset

        if not train_chck:
            canv.set_visible(env.get_local_matrix.copy(), env.actor_pos, [])

        score = 0
        while not done:
            if not train_chck:
                canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath)

            action = agent.greedy_epsilon(observation)
            observation_, reward, done, info = env.step(action, score)

            score += reward

            agent.store_transition(observation, observation_, reward, action,
                                   int(done))

            agent.learn()
            observation = observation_
        plt.data_in(score, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr, visit_cntr=env.visit_cntr)
        print(f'Ep {i}, score {score}, avg {plt.scores_avg[-1]}, epsilon {agent.epsilon}, lr {lr}')
        print(f'    Stayed {env.stay_cntr} : Walls {env.wall_cntr}')
        # Save NN every 10 its
        if i > 10 and i % 10 == 0:
            agent.save_models()
            plt.live_plot()