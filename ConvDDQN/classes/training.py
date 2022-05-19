from ConvDDQN.classes.canvas import Canvas
from ConvDDQN.classes.plotter import Plotter

import numpy as np
import torch

'''Training function
        defines training starting at (1,1) every epsiode
'''
def pretrain(name, episodes, gamma, memsize, batch_size,
             env, agent, canv,
             canv_chck:int=0,
             beta_inc:int=0, replace_testnet:int=100, EP:str=''
             ):

    optimal_path = np.load('classes/optimal_path.npy')
    warm_up_idx = 2
    cntr = 0
    plt = Plotter(name)  # initialise the handler for plotting live statistics

    for i in range(episodes):
        '''Reset Paramerts and environment handler at the beginning of each episode
                (Reset canvas handler if active)
        '''
        done = False  # return to unterminated state
        start_idx = tuple(optimal_path[-warm_up_idx])
        block = tuple(optimal_path[-warm_up_idx-1])
        observation = env.reset(start_idx, len(optimal_path[-warm_up_idx:])) #, block=block)  # reset environment and return starting observation

        if canv_chck:  # resent canvas
            canv.set_visible(env.loc.copy(), env.actor_pos, [])
        hs = (torch.zeros(2, 1, 512).float().to(agent.q_eval.device))  # initialise the hidden state for LSTM
        score = 0  # reset score
        reward = 0  # reset total rewards
        average_action = []  # track the average action (can be use to modify exploration if individual is stuck)
        epsilon_hist = []

        avg_pathlen = [0]

        '''Run through each episode'''
        while not done:
            '''Fetch action from greedy epsilon'''
            action, acts, hs, rand = agent.greedy_epsilon(observation, hs)

            '''Update Canvas Handler if in use'''
            if canv_chck and (i + 1) % (canv_chck) == 0:
                canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath, acts.data.cpu().numpy(), action,
                          score, reward, env.step_cntr, i, env.wall_cntr, env.visit_cntr, env.obs2D,
                          agent.epsilon, agent.lr,
                          gamma=gamma,
                          score_split=env.get_split_score(), step_split=env.get_split_step(), rand=rand,
                          EP=EP, memsize=memsize, batchsize=batch_size)

            '''Step environment to return resulting observations (future observations in memory)'''
            observation_, reward, done, info = env.step(action, score, warmup=optimal_path)
            path = len(env.actorpath)  # determine new path length
            score += reward  # update episode score

            if env.stay_cntr % replace_testnet == 0:
                agent.replace_target_network()

            '''Store experience in memory'''
            agent.store_transition(observation, observation_, reward, action, int(done))

            '''Step Learning at after each action'''
            loss = agent.learn()
            observation = observation_  # set current episode for following step
            epsilon_hist.append(agent.epsilon)  # track the epsilong values

        agent.replace_target_network()
        agent.step_params(beta_inc, path, i, np.mean(avg_pathlen[-30:]))  # step active agent parameters

        avg_pathlen.append(path)

        plt.data_in(score, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr,
                    visit_cntr=env.visit_cntr, path_cntr=path, epsilon=np.mean(epsilon_hist))

        if i % replace_testnet == 0:
            agent.replace_target_network()
            plt.live_plot()
        if i % 100 == 0:
            agent.save_models()
            env.meta_environment.save_meta_experience()

        if env.actor_pos == (199,199):
            cntr += 1

        if cntr > 2:
            warm_up_idx += 2
            cntr = 0

        print(f'Ep {i}, {loss} score {score}, Path Len {path}')

'''Training function
        defines training starting at (1,1) every epsiode
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
        hs = (torch.zeros(2, 1, 64).float().to(agent.q_eval.device))  # initialise the hidden state for LSTM
        score = 0  # reset score
        reward = 0  # reset total rewards
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
                          score_split=env.get_split_score(), step_split=env.get_split_step(), rand=rand,
                          EP=EP, memsize=memsize, batchsize=batch_size)

            '''Step environment to return resulting observations (future observations in memory)'''
            observation_, reward, done, info = env.step(action, score)

            path = len(env.actorpath)  # determine new path length
            score += reward  # update episode score
            score_tracker.append(score)

            '''Store experience in memory'''
            agent.store_transition(observation, observation_, reward, action, int(done))

            '''Step Learning at after each action'''
            loss = agent.learn()
            loss_tracker.append(loss)

            observation = observation_  # set current episode for following step

            inc_grad = 0  # tune this parameter to increase exploration if stuck
            epsilon_hist.append(agent.epsilon)  # track the epsilong values

            agent.step_params(beta_inc, env.step_cntr, i, np.mean(avg_pathlen[-30:]))  # step active agent parameters

            if env.step_cntr % replace_testnet == 0:
                agent.replace_target_network()

        agent.replace_target_network()

        avg_pathlen.append(path)

        plt.data_in(score_tracker, loss_tracker, wall_cntr=env.wall_cntr, stay_cntr=env.stay_cntr,
                    visit_cntr=env.visit_cntr, path_cntr=path, epsilon=np.mean(epsilon_hist))

        if i % 50 == 0:
            plt.live_plot()
            print(f'Ep {i}, {loss} score {score}, Path Len {path} ')
        if i % 2 == 0:
            agent.save_models()
            env.meta_environment.save_meta_experience()

