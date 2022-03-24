import gym
import numpy as np
from duelingddqn import Agent, Environment
from canvas import Canvas

if __name__ == '__main__':
    canv = Canvas()

    env = Environment()

    num_games = 2
    load_checkpoint = False

    agent = Agent(gamma=0.99, epsilon=1.0, lr=5e-5,
                  input_dims=[18], n_actions=5, mem_size=1000000, eps_min=0.01,
                  batch_size=64, eps_dec=1e-3, replace=1000)

    if load_checkpoint:
        agent.load_models()

    scores, eps_history = [], []

    for i in range(num_games):
        done = False
        observation = env.reset
        print(observation)
        score = 0

        while not done:
            canv.set_visible(env.get_local_matrix, env.get_actor_pos)
            canv.step_canvas()

            action = agent.greedy_epsilon(observation)
            observation_, reward, done, info = env.step(action)


            score += reward

            agent.store_transition(observation, observation_, reward, action,
                                   int(done))

            agent.learn()
            observation = observation_



        scores.append(score)
        avg_sc = np.mean(scores[-100:])
        print(f'Ep {i}, score {score}, avg {avg_sc}, epsilon {agent.epsilon}')

        if i > 10 and i % 10 == 0:
            agent.save_models()

