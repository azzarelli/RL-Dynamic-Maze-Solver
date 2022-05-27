"""Training and Testing functions

    Here we provide the tresting and training function which are fully automated. No parameters within this files
    should ever change.
"""

from ConvDDQN.classes.canvas import Canvas
from ConvDDQN.classes.plotter import Plotter

import numpy as np
import torch

'''Training function
        defines training starting at (1,1) every epsiode
        Run through `episode` episodes of training, and track data using plotting handler and external list
        We are given the option to visualise the progress using a canvas but note this is 3x slower than without
        visual aid.
'''
def train(name, episodes, gamma, memsize, batch_size,
             env, agent, canv:Canvas=None,
             canv_chck:int=0,
             beta_inc:int=0, replace_testnet:int=100, EP:str=''
             ):
    plt = Plotter(name)  # initialise the handler for plotting live statistics

    for i in range(episodes):
        '''Reset Paramerts and environment handler at the beginning of each episode
                (Reset canvas handler if active)
        '''
        done = False  # return to unterminated state
        observation = env.reset()  # reset environment and return starting observation
        if canv_chck:  # resent canvas
            canv.set_visible(env.loc.copy(), env.actor_pos, [])

        hs = (torch.zeros(2, 1, 512).float().to(agent.q_eval.device))  # initialise the hidden state for LSTM
        score = 0  # reset score
        reward = 0  # reset total rewards

        '''Define trackers for plotting'''
        average_action = []  # track the average action (can be use to modify exploration if individual is stuck)
        epsilon_hist = []
        loss_tracker = []
        score_tracker = []

        avg_pathlen = [0]

        '''Run through each episode'''
        while not done:
            '''Fetch action from greedy epsilon'''
            action, acts, hs, rand = agent.greedy_epsilon(observation, hs)
            average_action.append(float(action))

            '''Update Canvas Handler if in use'''
            if canv_chck and (i + 1) % (canv_chck) == 0:
                canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts.data.cpu().numpy(), action,
                          score, reward, env.step_cntr, i, env.wall_cntr, env.visit_cntr, env.obs2D,
                          agent.epsilon, agent.lr,
                          gamma=gamma,
                          step_split=env.get_split_step(), rand=rand,
                          EP=EP, memsize=memsize, batchsize=batch_size)

            '''Step environment to return resulting observations (future observations in memory)'''
            observation_, reward, done, info = env.step(action, score)
            path = len(env.actorpath)  # determine new path length
            score += reward  # update episode score
            score_tracker.append(score)

            '''Store experience in memory'''
            agent.store_transition(observation, observation_, reward, action, int(done))

            '''Step Learning of Network at after each action'''
            loss = agent.learn()
            loss_tracker.append(loss)

            observation = observation_  # set current episode for following step

            epsilon_hist.append(agent.epsilon)  # track the epsilong values

            agent.step_params(beta_inc, env.step_cntr, i, np.mean(avg_pathlen[-30:]))  # step active agent parameters

            '''Replace the target network every 100 transitions'''
            if env.step_cntr % replace_testnet == 0:
                agent.replace_target_network()

        avg_pathlen.append(path) # append to list-tracked

        '''Update plotting handler'''
        plt.data_in(score_tracker, loss_tracker, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr,
                    visit_cntr=env.visit_cntr, path_cntr=path, epsilon=np.mean(epsilon_hist))

        '''Plot results every 20 episodes to save time'''
        if i % 20 == 0:
            plt.live_plot()
            print(f'Ep {i}, {loss} score {score}, Path Len {path} ')
        '''Save models & meta-state information every two epsidoes'''
        if i % 2 == 0:
            agent.save_models()
            env.meta_environment.save_meta_experience()

'''Testing function

Note: This function REQUIRES the canvas! - for visualisation of the results.
'''
def test(gamma, memsize, batch_size,
        env, agent, canv:Canvas=None, EP:str=''
        ):
    for i in range(1):
        '''Reset Paramerts and environment handler at the beginning of each episode
                (Reset canvas handler if active)
        '''
        done = False  # return to unterminated state
        observation = env.reset()  # reset environment and return starting observation
        agent.epsilon = 0.
        canv.set_visible(env.loc.copy(), env.actor_pos, [])

        hs = (torch.zeros(2, 1, 512).float().to(agent.q_eval.device))  # initialise the hidden state for LSTM
        score = 0  # reset score
        reward = 0  # reset total rewards

        '''Trackers for testing output filer'''
        path_length = []
        time_step = []
        position = []
        local_observation = []
        action_taken = []

        '''Run through each episode'''
        while not done:
            '''Fetch action from greedy epsilon'''
            action, acts, hs, rand = agent.greedy_epsilon(observation, hs)

            '''Update Canvas Handler if in use'''
            canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts.data.cpu().numpy(), action,
                          score, reward, env.step_cntr, i, env.wall_cntr, env.visit_cntr, env.obs2D,
                          agent.epsilon, agent.lr,
                          gamma=gamma, step_split=env.get_split_step(), rand=rand,
                          EP=EP, memsize=memsize, batchsize=batch_size)

            action_taken.append(action)
            local_observation.append(env.loc)
            '''Step environment to return resulting observations'''
            observation_, reward, done, info = env.step(action, score)
            observation = observation_  # set current episode for following step

            path_length.append(len(env.actorpath))
            time_step.append(env.step_cntr)
            position.append(env.actor_pos)

    output_data = {"Position":position, "Step":time_step, "Action":action_taken,"Observations":local_observation, "Path Length":path_length}