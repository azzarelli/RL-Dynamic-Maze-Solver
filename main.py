import numpy as np
import matplotlib.pyplot as plt

from classes.agent import Agent
from classes.environment import Environment
from classes.canvas import Canvas

if __name__ == '__main__':
    #canv = Canvas()

    env = Environment()

    num_games = 2000
    load_checkpoint = False


    eps = 1.
    eps_dec = 1e-5
    if load_checkpoint:
        eps = 0.2
        eps_dec = 1e-5


    agent = Agent(gamma=0.99, epsilon=eps, lr=0.001,
                  input_dims=[19], n_actions=5, mem_size=1000000, eps_min=0.1,
                  batch_size=128, eps_dec=eps_dec, replace=1000, name='DDQN-full-ip18.pt')

    if load_checkpoint:
        agent.load_models()



    scores, scores_avg, eps_history = [], [], []


    for i in range(num_games):
        done = False
        observation = env.reset
        score = 0

        while not done:
            #canv.step_canvas(env.get_local_matrix, env.actor_pos, env.step_cntr)
            action = agent.greedy_epsilon(observation)
            observation_, reward, done, info = env.step(action)


            score += reward

            agent.store_transition(observation, observation_, reward, action,
                                   int(done))

            agent.learn()
            observation = observation_



        scores.append(score)
        score_avg = np.mean(scores[-100:])
        scores_avg.append(score_avg)
        print(f'Ep {i}, score {score}, avg {score_avg}, epsilon {agent.epsilon}')

        # Save NN every 10 its
        if i > 10 and i % 10 == 0:
            agent.save_models()
            plt.figure(1)
            plt.plot(scores)
            plt.xlabel('Epochs')
            plt.ylabel('Score')
            plt.savefig('liveplot/score.png')

            plt.figure(2)
            plt.plot(scores_avg)
            plt.xlabel('Epochs')
            plt.ylabel('Avg Score')
            plt.savefig('liveplot/avg_score.png')