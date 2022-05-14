"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
from old_models.ConvDoubleDQN.lib.read_maze import get_local_maze_information
import numpy as np
from PIL import Image
from PIL import ImageOps

from torchvision.transforms import transforms
from torchvision.utils import save_image

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
               "towards": +.0,
               "away":+.0,
               "visited":-0.,
               "wall":-0.0,
               "stay":-0.0,
               "bonus":+1.,
               "end":-0.
              }


class Environment:
    def __init__(self, img_size:int=10, multi_frame:bool=True):
        self.multiple_frames = multi_frame
        self.window_size = img_size
        self.visible_size = 16
        self.reset

    @property
    def reset(self):
        self.stay = 0
        self.step_cntr = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0
        self.score = 0
        self.prior_scores = []

        self.moved_max = 0
        self.bonus_taken = 0
        self.init = 0

        self.direction = 'stay'

        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]

        self.loc = []

        self.observation_map = np.array([[[0,0,0] for i in range(200)] for j in range(200)])
        self.img_history = []
        self.pos_history = []
        self.hist_idx = 0
        self.last_hist_idx = 0

        self.obs2D = []
        self.observation = self.observe_environment

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

    def render_global_map(self, loc):
        x, y = self.actor_pos

        pos = [(x + j - 1, y + i - 1) for i in range(3) for j in range(3)]
        for a, b in pos:  # update new info at new index
            i, j = a - x + 1, b - y + 1  # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0:  # wall
                self.observation_map[b][a] = [50, 50, 50]
            else:  # path
                self.observation_map[b][a] = [255, 255, 255]

        # Convert actor path to colour of global map
        for a, b in self.actorpath:
            self.observation_map[b][a] = [100, 255, 100]

        self.observation_map[y][x] = [255, 100, 100]
        self.observation_map[198][198] = [255, 255, 200]

    @property
    def observe_environment(self):
        x, y = self.actor_pos
        loc = get_local_maze_information(y, x)
        self.loc = loc.copy()
        if self.actor_pos in self.pos_history:
            idx = self.pos_history.index(self.actor_pos)
            self.observation_map = self.img_history[idx-1].copy()

            self.img_history = self.img_history[:idx+1]
            self.pos_history = self.pos_history[:idx+1]

            self.actorpath = self.actorpath[:idx+1]
            self.moved_max = len(self.actorpath)
            self.render_global_map(loc)
        else:
            self.render_global_map(loc)

            self.pos_history.append(self.actor_pos)
            self.img_history.append(self.observation_map.copy())


        if self.actor_pos not in self.actorpath and self.stay == 0:
            self.actorpath.append(self.actor_pos)
        self.stay = 0

        diff = int((self.window_size-1)/2)
        x_lb, x_ub = x-diff, x+diff
        y_lb, y_ub = y-diff, y+diff

        obs = [[[0, 0, 0] for i in range(self.window_size)] for j in range(self.window_size)]

        for j, y_i in enumerate(range(y_lb, y_ub+1)):
            for i, x_i in enumerate(range(x_lb, x_ub+1)):
                v = self.visible_size
                if y_i < 0 or x_i < 0:
                    obs[j][i] = [50, 50, 50]
                elif (x_i >= x -v and x+v >= x_i) and (y_i >= y-v and y+v >= y_i):
                    obs[j][i] = self.observation_map[y_i][x_i]
                else:
                    pass

        # obs = self.get_digits(obs, '6digitposition')
        # obs = self.get_digits(obs, '4digitscore')
        # obs = self.get_direction(obs)
        obsv_ = np.array(obs, dtype=np.uint8)
        img = Image.fromarray(obsv_, 'RGB')

        show_img = 0
        if show_img:
            img.save('obs_col')

        show_split = 1
        if show_split:
            im = self.transform(img)
            save_image(im[0], 'imgR.png')
            save_image(im[1], 'imgG.png')
            save_image(im[2], 'imgB.png')

        # reset environment so reload last 5 observations:
        if self.multiple_frames == True:

            self.obs2D = np.array(img)
            img = ImageOps.grayscale(img)
            # img.save('observationmap_grey.png')

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

            self.obs2D = np.array(img)
            # img.save('observationmap_col.png')

            self.obs = img
            imgs = self.transform(img)

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

        x,y, = self.actor_pos
        new_pos =  (x + x_inc, y + y_inc)
        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position
        obsv_mat = self.loc  # get prior position
        if action_dir[act_key]['id'] == 'stay':
            self.stay = 1
            self.stay_cntr += 1
        elif new_pos in self.actorpath:
            self.visit_cntr += 1
        elif obsv_mat[y_loc][x_loc][0] == 0:
            self.wall_cntr += 1

        paths = 0
        for i,o in enumerate(obsv_mat):
            for j,s in enumerate(o):
                if [i,j] in [[0,1],[1,0],[2,1],[1,2]]:
                    if s[0] != 0:
                        paths += 1
        if paths == 1:
            return self.observe_environment, -1000., True, {}  # terminate

        # # If too much time elapsed you die in maze :( (terminate maze at this point)
        # if self.step_cntr > 3500: #or
        #     print('I became an old man and dies in this maze...')
        #     return self.observe_environment, -0., True, {} # terminate

        if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
            return self.observe_environment, rewards_dir['stay'], False, {}
        if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
            return self.observe_environment, rewards_dir['wall'], False, {} # walking into walls is fatal

        self.actor_pos = new_pos
        if new_pos in self.actorpath:
            self.moved_max = len(self.actorpath)-1
            score = self.prior_scores[-1]
            self.prior_scores.pop(-1)
            return self.observe_environment, -score, False, {}

        # Have we reached the end?
        if self.actor_pos == (199, 199):
            return self.observation, 100000., True, {}

        #inclination to move in positive directions
        if x_inc > 0 or y_inc > 0:
            score = (1 - len(self.actorpath) / 3500)*2.
        else:
            score = 1 - len(self.actorpath) / 3500

        self.prior_scores.append(score)
        return self.observe_environment, score, False, {}
