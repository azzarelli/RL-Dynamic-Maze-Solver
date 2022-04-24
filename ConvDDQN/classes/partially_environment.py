"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
from DDQN.lib.read_maze import get_local_maze_information
import numpy as np
from PIL import Image
from PIL import ImageOps

from torchvision.transforms import transforms
import torch as T

global digit_dir
digit_dir = {'0':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
             '1':[[0,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0]],
             '2':[[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,1,0,0,0],[0,1,1,1,0],[0,0,0,0,0]],
             '3':[[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
             '4':[[0,0,0,0,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]],
             '5':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
             '6':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
             '7':[[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]],
             '8':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
             '9':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,0]]}

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
rewards_dir = {"newmax":+0.,
               "towards": +.0001,
               "away":+.0,
               "visited":-.1,
               "wall":-.1,
               "stay":-.1,
               "bonus":+5.,
               "end":-300.
              }


class Environment:
    def __init__(self, img_size:int=10, multi_frame:bool=True):
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0
        self.score = 0

        self.direction = 'stay'
        self.multiple_frames = multi_frame

        self.moved_max = 0
        self.init = 0
        self.bonus_taken = 0

        self.window_size = img_size
        self.visible_size = 10
        self.img_history = []

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
        self.score = 0

        self.moved_max = 0
        self.bonus_taken = 0
        self.init = 0

        self.direction = 'stay'

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.loc = []
        self.img_history = []

        self.observation_map = [[[50,50,50] for i in range(200)] for j in range(200)]

        self.observation = self.observe_environment
        self.obs2D = []

        return self.observation

    transform = transforms.Compose([transforms.ToTensor()])

    def get_split_score(self):
        global rewards_dir
        vis = self.visit_cntr*rewards_dir['visited']
        stay = self.stay_cntr*rewards_dir['stay']
        wall = self.wall_cntr*rewards_dir['wall']
        return [stay, vis, wall]

    def get_split_step(self):
        return [self.stay_cntr, self.visit_cntr, self.wall_cntr]

    def get_direction(self, obs):
        x,y = self.actor_pos
        if self.direction == 'stay':
            pass
        elif self.direction == 'up':
            for i in range(3):
                obs[2][17+i] = [255, 255, 255]
        elif self.direction == 'down':
            for i in range(3):
                obs[35][17+i] = [255, 255, 255]
        elif self.direction == 'left':
            for i in range(3):
                obs[17+i][2] = [255, 255, 255]
        elif self.direction == 'right':
            for i in range(3):
                obs[17+i][35] = [255, 255, 255]
        return obs

    def get_digits(self, obs, getType):
        x,y = self.actor_pos
        if getType == '6digitposition':
            # add position to image
            global action_dir
            #            hunderds | tens  | unit
            x_digit_num = (int(x / 100), int(x / 10) % 10, x % 10)
            y_digit_num = (int(y / 100), int(y / 10) % 10, y % 10)
            x_digits, y_digits = [], []
            for x_i, y_i in zip(x_digit_num, y_digit_num):
                x_d = np.array(digit_dir[str(x_i)])
                y_d = np.array(digit_dir[str(y_i)])
                x_digits.append(x_d)
                y_digits.append(y_d)

            x_digits = np.concatenate(x_digits, axis=1)
            y_digits = np.concatenate(y_digits, axis=1)
            pos_digits = np.concatenate((x_digits, y_digits), axis=1)
            a, b = 4, 25
            for j, pos_y in enumerate(pos_digits):
                for i, pos in enumerate(pos_y):
                    if pos == 1:
                        obs[b + j][a + i] = [255, 255, 255]

        elif getType == '4digitscore':
            path_len = int(self.score) #len(self.actorpath)
            p_digit_num = (int(path_len / 1000), int(path_len / 100) % 10, int(path_len / 10) % 10, path_len % 10)
            p_digits = []
            for p_i in p_digit_num:
                p_d = np.array(digit_dir[str(p_i)])
                p_digits.append(p_d)
            p_digits = np.concatenate(p_digits, axis=1)
            a, b = 4, 4
            for j, pos_y in enumerate(p_digits):
                for i, pos in enumerate(pos_y):
                    if pos == 1:
                        obs[b + j][a + i] = [255, 255, 255]

        elif getType == '4digitpath':
            path_len = len(self.actorpath)
            p_digit_num = (int(path_len / 1000), int(path_len / 100) % 10, int(path_len / 10) % 10, path_len % 10)
            p_digits = []
            for p_i in p_digit_num:
                p_d = np.array(digit_dir[str(p_i)])
                p_digits.append(p_d)
            p_digits = np.concatenate(p_digits, axis=1)
            a, b = 4, 4
            for j, pos_y in enumerate(p_digits):
                for i, pos in enumerate(pos_y):
                    if pos == 1:
                        obs[b + j][a + i] = [255, 255, 255]

        return obs

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
            self.observation_map[b][a] = [0, 255, 0]

        self.observation_map[y][x] = [255, 0, 0]
        self.observation_map[198][198] = [255, 255, 255]

        if self.actor_pos not in self.actorpath:
            self.actorpath.append(self.actor_pos)


        diff = int((self.window_size-1)/2)
        x_lb, x_ub = x-diff, x+diff
        y_lb, y_ub = y-diff, y+diff

        obs = [[[50, 50, 50] for i in range(self.window_size)] for j in range(self.window_size)]

        for j, y_i in enumerate(range(y_lb, y_ub+1)):
            for i, x_i in enumerate(range(x_lb, x_ub+1)):
                v = self.visible_size
                if y_i < 0 or x_i < 0:
                    pass
                elif (x_i >= x -v and x+v >= x_i) and (y_i >= y-v and y+v >= y_i):
                    obs[j][i] = self.observation_map[y_i][x_i]
                else:
                    pass # obs[j][i] = self.observation_map[y_i][x_i]

                # else:
                #     obs[j][i] = self.observation_map[y_i][x_i]

        # obs = self.get_digits(obs, '6digitposition')
        # obs = self.get_digits(obs, '4digitscore')
        # obs = self.get_direction(obs)
        obsv_ = np.array(obs, dtype=np.uint8)

        # reset environment so reload last 5 observations:
        if self.multiple_frames == True:
            img = Image.fromarray(obsv_, 'RGB')
            # img.save('observationmap.png')

            self.obs2D = np.array(img)
            img = ImageOps.grayscale(img)
            # img.save('observationmap_grey.png')
            self.img_history.append(img)

            if self.init == 0:
                self.obs = [img for i in range(4)]
                imgs = T.stack([self.transform(o) for o in self.obs], dim=1)[0]
                self.init += 1
            else:
                if img != self.obs[0]:
                    self.obs.insert(0, img)
                    self.obs.pop(-1)
                imgs = T.stack([self.transform(o) for o in self.obs], dim=1)[0]
        else:
            img = Image.fromarray(obsv_, 'RGB')
            # img.save('observationmap.png')
            self.img_history.append(img)

            self.obs2D = np.array(img)
            # img.save('observationmap_col.png')

            self.obs = img
            imgs = self.transform(img)

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
        self.score = score
        self.step_cntr += 1 # increment time
        global action_dir # Fetch action directory containing the properties of each action w.r.t environment
        act_key = str(action)
        global rewards_dir # Fetch reward directory

        x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)
        self.direction = action_dir[act_key]['id']

        x,y, = prior_pos = self.actor_pos
        new_pos =  (x + x_inc, y + y_inc)
        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position
        obsv_mat = self.loc  # get prior position

        if new_pos in self.actorpath:
            self.visit_cntr += 1
        if action_dir[act_key]['id'] == 'stay':
            self.stay_cntr += 1
        if obsv_mat[y_loc][x_loc][0] == 0:
            self.wall_cntr += 1

        # If too much time elapsed you die in maze :( (terminate maze at this point)
        if self.step_cntr > 4000: #or
            print('I became an old man and dies in this maze...')
            return self.observe_environment, -0., True, {} # terminate

        # Only allow penalisations a max of 600 times
        if self.visit_cntr > 0: #self.wall_cntr + self.stay_cntr + self.visit_cntr > 0: # 9*len(self.actorpath)/9:
            print('Visited Timeout')
            return self.observe_environment,  rewards_dir['end'], True, {}  # terminate


        if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
            return self.observe_environment, rewards_dir['stay'], False, {}
        if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
            return self.observe_environment, rewards_dir['wall'], False, {} # walking into walls is fatal
        if new_pos in self.actorpath:
            # if self.visit_cntr > 0: # we dont want to revisit prior positions
            #     return self.observe_environment, rewards_dir['end'], True, {}
            return self.observe_environment, rewards_dir['visited'], False, {}

        self.actor_pos = new_pos  # new global position if we move into a free space

        # Have we reached the end?
        if self.actor_pos == (199, 199):
            return self.observation, 100000., True, {}

        if (len(self.actorpath) + 1) % 5 == 0 and self.bonus_taken == 0:
            self.bonus_taken = 1
            print('Granted bonus ', str(len(self.actorpath)))
            return self.observe_environment, rewards_dir['bonus'], False, {}

        self.bonus_taken = 0
        if x_inc > 0 or y_inc > 0:
            return self.observe_environment, rewards_dir['towards'], False, {}

        # finally our only choice is to move away from goal
        return self.observe_environment, rewards_dir['away'], False, {}



    # def step(self, action, score):
    #     """Sample environment dependant on action which has occurred
    #
    #     Action Space
    #     ------------
    #     0 - no move
    #     1 - up
    #     2 - left
    #     3 - down
    #     4 - right
    #
    #     """
    #     self.score = score
    #     self.step_cntr += 1 # increment time
    #     global action_dir # Fetch action directory containing the properties of each action w.r.t environment
    #     act_key = str(action)
    #     global rewards_dir # Fetch reward directory
    #
    #     x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)
    #
    #     self.direction = action_dir[act_key]['id']
    #
    #     # If too much time elapsed you die in maze :( (terminate maze at this point)
    #     if self.step_cntr > 4000: #or
    #         print('I became an old man and dies in this maze...')
    #         return self.observe_environment, -0., True, {} # terminate
    #     # If we spent too long vising places we have already been
    #     if self.visit_cntr + self.wall_cntr > 600:
    #         print('Visisted Timeout')
    #         return self.observe_environment, -1., True, {}  # terminate
    #
    #     obsv_mat = self.loc # get prior position
    #     x, y = self.actor_pos
    #
    #     x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position
    #     if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
    #         self.stay_cntr += 1
    #         return self.observe_environment, rewards_dir['stay'], False, {}
    #
    #     if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
    #         self.wall_cntr += 1
    #         return self.observe_environment, rewards_dir['wall'], False, {} # walking into walls is fatal
    #
    #     # nmax = np.sqrt(((x+x_inc)**2+(y+y_inc)**2))
    #     # if nmax > self.moved_max:
    #     #     self.moved_max = nmax
    #     #     self.actor_pos = new_pos = (x + x_inc, y + y_inc)  # new global position if we move into a free space
    #     #
    #     #     return self.observe_environment, rewards_dir['newmax'], False, {}
    #
    #     # So if we do successfully move
    #     self.actor_pos = new_pos = (x + x_inc, y + y_inc) # new global position if we move into a free space
    #
    #     # # Have we visited this spot already?
    #     if self.actor_pos in self.actorpath:
    #         return self.observe_environment, rewards_dir['visited'], False, {}
    #
    #     # Have we reached the end?
    #     if new_pos == (199, 199):
    #         return self.observation, 100., True, {}
    #
    #     if x_inc > 0 or y_inc > 0:
    #         return self.observe_environment, rewards_dir['towards'], False, {}
    #
    #     # finally our only choice is to move away from goal
    #     return self.observe_environment, rewards_dir['away'], False, {}