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
from PIL import Image
from PIL import ImageOps
from matplotlib import cm

from torchvision.transforms import transforms
import torch as T
import torchvision.transforms as tv

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
               "away":+1.,
              "visited":-1.,
              "wall":-1.,
              "stay":-1.
              }


class Environment:
    def __init__(self, img_size:int=10):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.init = 0

        self.window_size = img_size

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.observation_map = [[[50,50,50] for i in range(200)] for j in range(200)]

        self.obs2D = []
        self.loc = []
        self.observation = self.observe_environment


    @property
    def reset(self):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0

        self.init = 0

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.loc = []
        self.observation_map = [[[50,50,50] for i in range(200)] for j in range(200)]

        self.observation = self.observe_environment
        self.obs2D = []

        return self.observation

    transform = transforms.Compose([transforms.ToTensor()])

    @property
    def observe_environment(self):
        x, y = self.actor_pos
        loc = get_local_maze_information(y, x)
        self.loc = loc.copy()

        pos = [(x + j - 1, y + i -1) for i in range(3) for j in range(3)]
        for a, b in pos: # update new info at new index
            i, j = a-x+1, b-y+1 # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0: # wall
                self.observation_map[b][a] = [0, 0, 0]
            else: # path
                self.observation_map[b][a] = [255, 255, 255]

        # Convert actor path to colour of global map
        for a,b in self.actorpath:
            self.observation_map[b][a] = [100, 100, 100]

        self.observation_map[y][x] = [255, 0, 0]
        self.observation_map[198][198] = [0, 0, 255]
        if self.actor_pos not in self.actorpath:
            self.visit_cntr = 0
            self.actorpath.append(self.actor_pos)
        else:
            self.visit_cntr += 1

        diff = int((self.window_size-1)/2)
        x_lb, x_ub = x-diff, x+diff
        y_lb, y_ub = y-diff, y+diff

        obs = [[[50, 50, 50] for i in range(self.window_size)] for j in range(self.window_size)]

        for j, y_i in enumerate(range(y_lb, y_ub+1)):
            for i, x_i in enumerate(range(x_lb, x_ub+1)):
                if y_i < 0 or x_i < 0:
                    pass
                else:
                    obs[j][i] = self.observation_map[y_i][x_i]

        obsv_ = np.array(obs, dtype=np.uint8)

        img = Image.fromarray(obsv_, 'RGB')
        img.save('observationmap.png')

        self.obs2D = np.array(img)
        img = ImageOps.grayscale(img)

        # reset environment so reload last 5 observations:
        if self.init == 0:
            self.obs = [img for i in range(2)]
            imgs = T.stack([self.transform(o) for o in self.obs], dim=1)[0]
            self.init += 1
        else:
            self.obs.insert(0, img)
            self.obs.pop(-1)
            imgs = T.stack([self.transform(o) for o in self.obs], dim=1)[0]


        # img = tv.Grayscale()(img)
        # img = img.resize((400,400))

        # img = self.transform(img)
        # img = tv.Pad(padding=50)(img)
        # imgnpy = img.numpy()
        # img = T.from_numpy(imgnpy[0])
        # print(img)

        # img = self.transform(img)
        return imgs

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
        if self.step_cntr > 4000 or self.step_cntr > len(self.actorpath)*3: #or
            print('I became an old man and dies in this maze...')
            return self.observe_environment, -0., True, {} # terminate
        # If we spent too long vising places we have already been
        # if self.visit_cntr > 50:
        #     print('Visisted Timeout')
        #     return self.observe_environment, -2., True, {}  # terminate

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
