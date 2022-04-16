"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
import time
from lib.read_maze import get_local_maze_information
import numpy as np

# We set the action-space directory to access
global action_dir
action_dir = {"0": {"id":'stay',
                    "move":(0,0)},
              "1": {"id":'up',
                    "move":(0,-1)},
              "2": {"id":'left',
                    "move":(-1,0)},
              "3": {"id":"down",
                    "move":(0,1)},
              "4": {"id":'right',
                    "move":(1,0)}
              }

global rewards_dir
rewards_dir = {"move": -.04,
              "visited":-1.,
              "wall":-1.,
              "stay":-1.,
              "deadend":-200.,
               "oldman":-100.,
              }


class Environment:
    def __init__(self):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.observation = self.observe_environment  # set up empty state
        self.obs2D = []



    @property
    def reset(self):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]

        self.observation = self.observe_environment  # set up empty state
        return self.observation

    @property
    def observe_environment(self):
        x, y = self.actor_pos
        loc = get_local_maze_information(y, x)
        self.obs2D = loc.copy()
        l1, l2, l3 = [], [], []
        for l in range(len(loc)):
            for j in range(len(loc[l])):
                x_ = x + j - 1
                y_ = y + l - 1

                if loc[l][j][0] == 0:
                    l1.append(0)
                else:
                    l1.append(1)

                if loc[l][j][0] == 0:
                    l2.append(0)
                elif (x_, y_) in self.actorpath:
                    l2.append(0)
                else:
                    l2.append(1)


        self.observation = l1 + l2 + [self.actor_pos[0], self.actor_pos[1]] # , self.step_cntr ]# + [prior_pos[0], prior_pos[1]]

        if self.actor_pos not in self.actorpath:
            self.visit_cntr = 0
            self.actorpath.append(self.actor_pos)
        else:
            self.visit_cntr += 1

        return self.observation

    @property
    def get_local_matrix(self):
        """Return observation matrix from 1-D observation data

        Notes
        -----
        Observation is a 1-Dim vector with wall and fire data in alternating index values
        self.observation = [w_00, f_00, w_10, f_10, ..., w_2_2, f_22] for w_rc, f_rc, where r - rows and c - cols
        (note r, c is called as `obj[r][c]` in class-method `step`)

        We return the observation matrix rather than 1-D array to facilitate calling the observation within the environment
        i.e. accessing through `[r][c]` is much easier than cycling through `obj[i]`
        """
        return self.obs2D

    @property
    def get_actor_pos(self):
        return self.actor_pos

    def step(self, action, score):
        """Sample environment dependant on action which has occurred

        Action Space
        ------------
        0 - no move
        1 - up
        2 - left
        3 - down
        4 - right

        """
        self.step_cntr += 1 # increment time
        global action_dir # Fetch action directory containing the properties of each action w.r.t environment
        act_key = str(action)
        global rewards_dir # Fetch reward directory

        x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)

        # If too much time elapsed you die in maze :( (terminate maze at this point)
        if self.step_cntr > 600:
            print('I became an old man and dies in this maze...')
            return self.observe_environment, rewards_dir['oldman'], True, {} # terminate
        # If we spent too long vising places we have already been
        # if self.visit_cntr > 50:
        #     print('Visisted Timeout')
        #     return self.observe_environment, -2., True, {}  # terminate

        obsv_mat = self.get_local_matrix # get prior position
        x, y = self.actor_pos

        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position

        if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
            self.stay_cntr += 1
            return self.observe_environment, rewards_dir['stay'], False, {}

        if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
            self.wall_cntr += 1
            return self.observe_environment, rewards_dir['wall'], False, {} # walking into walls is fatal

        # # Check to see if we are either blocked in as a result of prior path, wall or fire (if so -> reward staying)
        # is_blocked = False
        # path = 0
        # for i, o in enumerate(obsv_mat):
        #     for j, p in enumerate(o):
        #         if (i,j) in [(0,1), (1,0), (1,2), (2,1)]:
        #             pos = (x + j - 1, y + i - 1)
        #             if p[0] == 1:
        #                 path += 1
        # if path == 1:
        #     is_blocked = True
        # if is_blocked : # Reward staying if path is blocked
        #     print('Deadend')
        #     return self.observe_environment, rewards_dir['deadend'], True, {}

        # So if we do successfully move
        self.actor_pos = new_pos = (x + x_inc, y + y_inc) # new global position if we move into a free space
        # Have we reached the end?
        if new_pos == (199, 199):
            return self.observation, 100., True, {}

        # Have we visited this spot already?
        if self.actor_pos in self.actorpath:
            return self.observe_environment, rewards_dir['visited'], False, {}

        # finally our only choice is to move away from goal
        return self.observe_environment, rewards_dir['move'] + (0.01 * float(self.actor_pos[0] + self.actor_pos[1])/2), False, {}
