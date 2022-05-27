"""

Notes
-----


"""

from Manual_Path_Finder.classes.partially_environment import Environment
from Manual_Path_Finder.classes.canvas import Canvas
import pygame

import numpy as np
fp = np.load('optimal_path.npy')
print(len(fp))


if __name__ == '__main__':
    '''Initialise Handlers for Environment & Agent '''
    env = Environment() # enviornment handler

    '''Initialise canvas if visualising training (note this method is 3x slower than training without canvas)'''
    canv = Canvas()

    done = False  # return to unterminated state
    observation = env.reset((1,1))  # reset environment and return starting observation
    canv.set_visible(env.loc.copy(), env.actor_pos, [])


    '''Run through each episode'''
    while not done:
        '''Fetch action from greedy epsilon'''
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action= 1
        elif keys[pygame.K_RIGHT]:
            action= 4
        elif keys[pygame.K_DOWN]:
            action= 3
        elif keys[pygame.K_LEFT]:
            action= 2
        else:
            action=0
        '''Update Canvas Handler if in use'''
        canv.step(env.obs2D.copy(), env.actor_pos, env.actorpath)

        '''Step environment to return resulting observations (future observations in memory)'''
        observation_, reward, done, info = env.step(action, 0)
