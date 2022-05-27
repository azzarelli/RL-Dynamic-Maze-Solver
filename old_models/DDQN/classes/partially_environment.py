"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
from old_models.DDQN.lib.read_maze import get_local_maze_information
import numpy as np
from PIL import Image
from PIL import ImageOps

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
rewards_dir = {"towards": +1.,
               "away":+.8,
              "visited":-.05,
              "wall":-.1,
              "stay":-0.01
              }


class Environment:
    def __init__(self):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.observation_map = [[-1 for i in range(200)] for j in range(200)]
        self.observation_size = 10
        self.not_observed = 0
        self.observation = self.observe_environment
        o = self.observation[:-3]  # set up empty state
        self.observations = [o[(i) * int(len(o) / 3):(i + 1) * int(len(o) / 3)] for i in range(3)]
        self.obs2D = []


    @property
    def reset(self):
        self.observation_size = 10

        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]

        self.observation_map = [[-1 for i in range(200)] for j in range(200)]
        self.not_observed = 0
        self.observation = self.observe_environment
        o = self.observation[:-3]  # set up empty state
        self.observations = [o[(i) * int(len(o) / 3):(i+1) * int(len(o) / 3)] for i in range(3)]

        return self.observation

    def observation_2dlarge(self, l1, name):
        obsv = [[[0, 0, 0] for j in range(self.observation_size)] for i in range(self.observation_size*3)]
        for i, l in enumerate(l1):
            c, d = i % self.observation_size, int(i / self.observation_size)
            if l == 1:
                obsv[d][c] = [0, 255, 0]
            elif l == 2:
                obsv[d][c] = [255, 255, 255]

            elif l == 0:
                obsv[d][c] = [0, 0, 0]
            else:
                obsv[d][c] = [0, 140, 50]

        obsv_ = np.array(obsv,  dtype=np.uint8)
        img = Image.fromarray(obsv_, 'RGB')
        img = ImageOps.grayscale(img)
        # Example of observation
        img.save('observation'+name+'.png')

        #img = self.transform(img)

        return obsv, img

    def observation_2d(self, l1, name):
        obsv = [[[0, 0, 0] for j in range(self.observation_size)] for i in range(self.observation_size)]
        for i, l in enumerate(l1):
            c, d = i % self.observation_size, int(i / self.observation_size)
            if l == 1:
                obsv[d][c] = [0, 255, 0]
            elif l == 2:
                obsv[d][c] = [255, 255, 255]

            elif l == 0:
                obsv[d][c] = [0, 0, 0]
            elif l == -1:
                obsv[d][c] = [0, 140, 50]
            else:
                obsv[d][c] = [0, 50, 50]

        obsv_ = np.array(obsv,  dtype=np.uint8)
        img = Image.fromarray(obsv_, 'RGB')
        img = ImageOps.grayscale(img)
        # Example of observation
        img.save('observation'+name+'.png')

        #img = self.transform(img)

        return obsv, img

    @property
    def observe_environment(self):
        x, y = self.actor_pos
        pos = [(x + j - 1, y + i -1) for i in range(3) for j in range(3)]

        loc = get_local_maze_information(y, x)
        self.loc = loc

        for a, b in pos: # update new info at new index
            i, j = a-x+1, b-y+1 # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0:
                self.observation_map[b][a] = 0
            else:
                self.observation_map[b][a] = 2

        # Convert actor path to colour of global map
        # for a,b in self.actorpath:
        #     self.observation_map[b][a] = 3
        self.observation_map[y][x] = 1

        if self.actor_pos not in self.actorpath:
            self.visit_cntr = 0
            self.actorpath.append(self.actor_pos)
        else:
            self.visit_cntr += 1

        c = int(self.observation_size / 2)

        a_1, a_2 = self.actor_pos[0] - c, self.actor_pos[0] + c
        b_1, b_2 = self.actor_pos[1] - c, self.actor_pos[1] + c

        # Fetch 21 x 21 local observation of map
        l1 = []
        for j in range(b_1, b_2):
            for i in range(a_1, a_2):

                if j < 0 or i < 0:
                    l1.append(-2)
                else:
                    l1.append(self.observation_map[j][i])

        # What the observation map sees
        # obsv = [[[0,0,0] for j in range(200)] for i in range(200)]
        # for j in range(len(self.observation_map)):
        #     for i in range(len(self.observation_map[j])):
        #         if self.observation_map[j][i] == 0:
        #             obsv[j][i] = [255,0, 0]
        #         elif self.observation_map[j][i] == -1:
        #             obsv[j][i] = [0,0,0]
        #
        #         elif self.observation_map[j][i] == 1:
        #             obsv[j][i] = [255,255, 255]
        #         elif self.observation_map[j][i] == -2:
        #             obsv[j][i] = [0,0,255]
        #         elif self.observation_map[j][i] == 2:
        #             obsv[j][i] = [0, 255, 0]

        #   What the 20x20 observation sees
        _, img = self.observation_2d(l1, '1')

        # Show bitmap
        # from matplotlib import pyplot as plt
        # plt.imshow(obsv, interpolation='nearest')
        # plt.show()
        if self.not_observed != 0:
            l2 = self.observations[0].copy()

            self.observations[0] = l1

            # only change prior observation when a change has occured
            if l2 != l1:
                self.observations.append(l2)
                self.observations.pop(1)
                # self.prior_observation = l2


            # l = self.observations[0] + self.observations[4] +self.observations[3] +self.observations[2]+\
            #     self.observations[1]
            l = self.observations[0] + self.observations[2] + self.observations[1]

            self.obs2D, img2 = self.observation_2dlarge(l, '2')

            self.observation = l + [self.actor_pos[0], self.actor_pos[1], len(self.actorpath)]
            return self.observation

        else:
            l2 = l1.copy()
            self.not_observed += 1
            l = l1 + l2 + l2
            self.obs2D, img2 = self.observation_2dlarge(l, '2')

            o = l + [self.actor_pos[0], self.actor_pos[1], len(self.actorpath)]
            return o

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
        if self.step_cntr > len(self.actorpath)*4 or self.step_cntr > 4000:
            print('I became an old man and dies in this maze...')
            return self.observe_environment, -0., True, {} # terminate
        # If we spent too long vising places we have already been
        # if self.visit_cntr > 50:
        #     print('Visisted Timeout')
        #     return self.observe_environment, -2., True, {}  # terminate


        if self.visit_cntr > 1 or self.wall_cntr > 1:
            print('Visisted Timeout')
            return self.observe_environment, -1., True, {}  # terminate


        obsv_mat = self.loc # get prior position
        x, y = self.actor_pos

        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position

        if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
            self.stay_cntr += 1
            return self.observe_environment, rewards_dir['stay'], False, {}

        if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
            self.wall_cntr += 1
            return self.observe_environment, rewards_dir['wall'], False, {} # walking into walls is fatal

        # So if we do successfully move
        self.actor_pos = new_pos = (x + x_inc, y + y_inc) # new global position if we move into a free space
        # Have we reached the end?
        if new_pos == (199, 199):
            return self.observation, 100., True, {}
        #
        # # Have we visited this spot already?
        if self.actor_pos in self.actorpath:
            return self.observe_environment, rewards_dir['visited'], False, {}

        if x_inc > 0 or y_inc > 0:
            return self.observe_environment, rewards_dir['towards'], False, {}

        # finally our only choice is to move away from goal
        return self.observe_environment, rewards_dir['away']*(len(self.actorpath)/10), False, {}
