"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.

"""

from ConvDDQN.lib.read_maze import get_local_maze_information
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

from torchvision.transforms import transforms
from torchvision.utils import save_image

import time

import torch as T

global digit_dir
digit_dir = {'none':[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
             '0':[[0,0,0,0,0],[0,1,1,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,1,1,0],[0,0,0,0,0]],
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

# Colours for Constructing images
WALL = [0, 0, 0]
PATH = [50, 30, 30]
EMPTYPATH = [255,255,255]
END = EMPTYPATH
ACTOR = [180, 180, 200]
FIRE = [30,50,30]
ACTORBLUR = [0, 0, 100]

optimal_path = np.load('classes/optimal_path.npy')

class Environment:
    def __init__(self, img_size:int=10, multi_frame:bool=True):
        self.multiple_frames = multi_frame
        self.window_size = img_size
        self.visible_size = 5
        self.reset()

        self.meta_environment = MetaEnvironment()

    def reset(self, start_position:tuple=(1,1), max_steps:int=3500, block:tuple=(0,0)):
        self.stay = 0
        self.step_cntr = 0
        self.step_since_move = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0
        self.score = 0
        self.prior_scores = []

        self.max_steps = 2*max_steps if max_steps > 100 else 100
        self.start_position = start_position
        self.block = block
        self.bonus_taken = 0
        self.init = 0

        self.direction = 'stay'

        self.actor_pos = start_position
        self.prior_position = [start_position,start_position]
        self.actorpath = [self.actor_pos]

        self.loc = []

        self.observation_map = np.array([[EMPTYPATH for i in range(201)] for j in range(201)])
        self.img_history = []
        self.pos_history = []
        self.hist_idx = 0
        self.last_hist_idx = 0

        self.obs2D = []
        self.observation = self.observe_environment()

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

    def render_global_map(self, loc):
        x, y = self.actor_pos
        pos = [(x + j - 1, y + i - 1) for i in range(3) for j in range(3)]
        for a, b in pos:  # update new info at new index
            i, j = a - x + 1, b - y + 1  # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0:  # wall
                self.observation_map[b][a] = WALL
            elif loc[j][i][1] > 0:  # wall
                self.observation_map[b][a] = FIRE
            else:  # path
                self.observation_map[b][a] = EMPTYPATH
            if (a,b) == self.block:
                self.observation_map[b][a] = WALL
                self.loc[j][i][0] = 0

        # Convert actor path to colour of global map
        for a, b in self.actorpath:
            self.observation_map[b][a] = PATH

        self.observation_map[y][x] = ACTOR
        self.observation_map[199][199] = END

    def render_local_state(self, vel):
        """Render the local state, which will be be processed for input into out DQN

        Notes
        -----
        To populate the image we take a snapshot of of the environment we have explored in the episode and reduce the
        window size so that we see a sub-map of the global map.
        """
        x, y = self.actor_pos # fetch the actors position
        last_position = (x, y)
        if vel > 1:
            last_position = self.prior_position[-1]


        '''Calculate range of visible to the agent in the local map
                (combines local observation with learnt environment)
        '''
        diff = int((self.window_size - 1) / 2)
        x_lb, x_ub = x - diff, x + diff
        y_lb, y_ub = y - diff, y + diff

        obs = [[[0, 0, 0] for i in range(self.window_size)] for j in range(self.window_size)] # initilise observation matrix
        obs_gray = [[0 for i in range(self.window_size)] for j in range(self.window_size)]
        for j, y_i in enumerate(range(y_lb, y_ub+1)):
            for i, x_i in enumerate(range(x_lb, x_ub+1)):
                v = self.visible_size # the perpendicular distance visible to our DQN agent

                # Observations which exist outside of the global map are processed, otherwise ignored
                if (x_i >= x-v and x+v >= x_i) and (y_i >= y-v and y+v >= y_i):
                    if y_i < 0 or x_i < 0 or y_i > 199 or x_i > 199:
                        obs[j][i] = WALL
                        obs_gray[j][i] = 0
                    else:
                        '''Append known pizel-states to enlarged local observation matrix'''
                        obs[j][i] = self.observation_map[y_i][x_i]

                        if np.array_equal(obs[j][i], WALL):
                            obs_gray[j][i] = 0
                        elif np.array_equal(obs[j][i], PATH):
                            obs_gray[j][i] = 50
                        elif np.array_equal(obs[j][i], ACTOR):
                            obs_gray[j][i] = 120
                        elif np.array_equal(obs[j][i], EMPTYPATH):
                            obs_gray[j][i] = 255
                        elif np.array_equal(obs[j][i], FIRE):
                            obs_gray[j][i] = 180

                        '''If we have a moving velocity we want to note the prior position'''
                        # if (x_i, y_i) == last_position and vel > 1:
                        #     obs[j][i] = ACTORBLUR # Choose between showing the blud on the actor - channel
                            #obs_gray[j][i] = 120 # or on the greyscale local environment (not both as we want to
                                                  # minimise the changes in state-space)

        return obs, obs_gray

    def observe_environment(self, velocity:int=1):
        """We fetch our knowledge from prior complete map of environment (shows everything we have currently seen)
        and return either a set of greyscale images or  a single coloured image as the partially observed state

        Notes
        -----
        Duelling DQN has difficulties with repeated states in the same episode. If our actor moves towards the objective
        and then away from the objective, the observation before moving towards and the observation after moving away
        should be the same - with the new knowledge of the step ahead this may not be the case as by moving back we
        now observe the path ahead - however this is not so good as Duelling DQN has trouble assigning future rewards
        to states which are repeated. If I repeat this action twice we want both the observations ad rewards as a result
        from the observations to be the same

        Other
        -----
        We render the game as a square view-port with a score on the top, separated by a white line and the game-space
        below. The game space is a nxn view of known pixel surrounding the actor (e.g. if n=5 we would see 2 pixels in
        direction from the actor), the caveat being, we only see walls/paths which have priorly been encountered - we
        don't see the global maze.
        """
        x, y = self.actor_pos # fetch the actors position
        loc = get_local_maze_information(y, x) # fetch the local observation at the lowest-level
        self.loc = loc
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

        '''Now the global state space has been updated, we can update the actor's path'''
        if self.actor_pos not in self.actorpath and self.stay == 0:
            self.actorpath.append(self.actor_pos)
        self.stay = 0

        '''Fetch RGB and Greyscale representations of local enlarged local statespace
                Notes - Enlarging visual region from 3x3 to nxn where n>3, means we don't run into seemingly repeated 
                        observtions for 3x3 blocks which are same but on different parts of the mase, with more info of 
                        environment states become more unique 
        '''
        obs, obs_gray = self.render_local_state(velocity)

        '''Image Processing for numpy array into suitable tensor'''
        obsv_ = np.array(obs, dtype=np.uint8)
        obs_gray = np.array(obs_gray, dtype=np.uint8)
        img = Image.fromarray(obsv_, 'RGB')
        gimg = Image.fromarray(obs_gray, 'L')
        img = img.resize((40, 40), resample=Image.NEAREST)


        imgs = self.transform(img)
        gimg = self.transform(gimg)

        self.obs2D = np.array(img) # save enlarged observations for other processes
        self.obs = img

        '''Manual Testing on image processing
                Channels:
                    R (0)    - Path trace
                    G (1)    - Maze walls
                    B (2)    - Actor and velocity indicator
                    Grey (3) - Greyscale of enlarged local observation
        '''
        # Choice to save images
        show_img = 0
        if show_img:
            save_image(imgs, 'obs_col.png') # save coloured obseervation
        show_channels = 0
        if show_channels:
            save_image(imgs[0], 'img_R.png') # track prior path
            save_image(imgs[1], 'img_G.png') # track
            save_image(imgs[2], 'img_B.png')
            save_image(gimg, 'gimg.png')

        '''Merge the color channels with greyscale (entire) local map'''
        #imgs = T.concat([imgs, gimg], dim=0)

        return imgs

    @property
    def get_actor_pos(self):
        return self.actor_pos

    def valid_movement(self, next_local_position):
        valid_move = True
        reason = ''
        x_new, y_new = next_local_position
        obsv_mat = self.loc  # get prior position

        '''Staying Put'''
        if next_local_position == (1,1):
            valid_move = False
            reason = 'stay'
        '''Wall Blocking'''
        if obsv_mat[y_new][x_new][0] == 0:
            valid_move = False
            reason = 'wall'
        '''Fire Blocking'''
        if obsv_mat[y_new][x_new][1] > 0:
            valid_move = False
            reason = 'fire'

        return valid_move, reason

    def movement_velocity(self, new_position):
        """Check if the actor is making consequent steps (or stopping in the middle of a walk)

        Notes
        -----
        Velocity is determined by whether the actor is moving away from initial target, if several consequent steps have
        been taken, velocity = 2 otherwise velocity = 1. This constant is used as a multiplicative factor increasing
        reward of movement depednant on velocity (higher rewards for more consequent steps)
        """
        velocity = 1
        if new_position not in self.actorpath: # check if we have moved away from start
            if self.prior_position[0] != self.prior_position[1]:
                velocity = 2
        return velocity

    def step(self, action, score, warmup=None):
        """Sample environment dependant on action which has occurred
        """
        self.score = score
        self.step_cntr += 1 # increment time
        self.step_since_move += 1

        global action_dir # Fetch action directory containing the properties of each action w.r.t environment
        act_key = str(action)
        global rewards_dir # Fetch reward directory

        x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)
        self.direction = action_dir[act_key]['id']

        x,y, = self.actor_pos
        self.prior_position.pop(0) # remove second last position
        self.prior_position.append((x,y)) # append new last position
        new_pos =  (x + x_inc, y + y_inc)

        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position
        obsv_mat = self.loc  # get prior position

        valid_move, reason = self.valid_movement((x_loc, y_loc))

        if not valid_move:
            self.stay = 1
            if reason == 'wall':
                self.wall_cntr += 1
            elif reason == 'stay':
                self.stay_cntr += 1
            elif reason == 'fire':

                pass # TODO : initialise and step fire counter

        elif new_pos in self.actorpath:
            self.visit_cntr += 1


        '''Dead-end leads to death
                Getting our agent out of deadend if too complex for this problem (deadend may be quite long), 
                it is much easier to force death so agent spend more time exploring other viable areas
        '''
        paths = 0
        for i,o in enumerate(obsv_mat):
            for j,s in enumerate(o):
                if [i,j] in [[0,1],[1,0],[2,1],[1,2]]:
                    if s[0] != 0:
                        paths += 1
        if paths == 1:
            #print('Path Death')
            return self.observe_environment(), -0., True, {}  # terminate

        '''Force Death if Episode too long''' # Fine if using Duelling DQN but may be problematic with Vanilla DQN
        if abs(self.step_cntr) > 8000:# 4000: #self.max_steps: #or
            #print('I became an old man and dies in this maze...')
            return self.observe_environment(), -0., True, {} # terminate

        '''Handle Invalid Movment'''
        if not valid_move:
            self.meta_environment.update_history(self.actor_pos) # update meta-environment
            return self.observe_environment(), -0, False, {}

        '''Valid Movement'''
        self.actor_pos = new_pos # set new position
        self.meta_environment.update_history(new_pos) # update meta-environment

        '''If we are re-tracing steps'''
        if new_pos in self.actorpath:
            self.moved_max = len(self.actorpath)-1
            score = self.prior_scores[-1]
            self.prior_scores.pop(-1)
            return self.observe_environment(), -score, False, {}

        # Have we reached the end?
        if self.actor_pos == (199, 199):
            print('Final achieved')
            return self.observe_environment(), 10000., True, {}

        velocity = self.movement_velocity(new_position=new_pos)

        score = 0
        if new_pos in optimal_path:
            score = (1- (len(self.actorpath) / 3600)) #/ self.step_since_move #* velocity

        self.prior_scores.append(score)

        self.step_since_move = 0 # reset
        return self.observe_environment(velocity), score, False, {}


class MetaEnvironment:
    def __init__(self):
        self.environment_history = np.zeros((200,200))

    def update_history(self, global_position):
        x,y = global_position
        self.environment_history[y][x] += 1

    def save_meta_experience(self, fp:str='META_experience.data'):
        np.save(fp, self.environment_history)
        plt.imshow(self.environment_history, cmap='hot', interpolation='nearest')
        plt.savefig('liveplot/METADATA.png')

        # def step(self, action, score, warmup=None):
        #     """Sample environment dependant on action which has occurred
        #     """
        #     self.score = score
        #     self.step_cntr += 1  # increment time
        #     self.step_since_move += 1
        #
        #     global action_dir  # Fetch action directory containing the properties of each action w.r.t environment
        #     act_key = str(action)
        #     global rewards_dir  # Fetch reward directory
        #
        #     x_inc, y_inc = action_dir[act_key]['move']  # fetch movement from position (1,1)
        #     self.direction = action_dir[act_key]['id']
        #
        #     x, y, = self.actor_pos
        #     self.prior_position.pop(0)  # remove second last position
        #     self.prior_position.append((x, y))  # append new last position
        #     new_pos = (x + x_inc, y + y_inc)
        #
        #     x_loc, y_loc = (1 + x_inc, 1 + y_inc)  # Update Local Position
        #     obsv_mat = self.loc  # get prior position
        #
        #     valid_move, reason = self.valid_movement((x_loc, y_loc))
        #
        #     if not valid_move:
        #         self.stay = 1
        #         if reason == 'wall':
        #             self.wall_cntr += 1
        #         elif reason == 'stay':
        #             self.stay_cntr += 1
        #         elif reason == 'fire':
        #
        #             pass  # TODO : initialise and step fire counter
        #
        #     elif new_pos in self.actorpath:
        #         self.visit_cntr += 1
        #
        #     '''Dead-end leads to death
        #             Getting our agent out of deadend if too complex for this problem (deadend may be quite long),
        #             it is much easier to force death so agent spend more time exploring other viable areas
        #     '''
        #     paths = 0
        #     for i, o in enumerate(obsv_mat):
        #         for j, s in enumerate(o):
        #             if [i, j] in [[0, 1], [1, 0], [2, 1], [1, 2]]:
        #                 if s[0] != 0:
        #                     paths += 1
        #     if paths == 1:
        #         # print('Path Death')
        #         return self.observe_environment(), -0., True, {}  # terminate
        #
        #     '''Force Death if Episode too long'''  # Fine if using Duelling DQN but may be problematic with Vanilla DQN
        #     if abs(self.step_cntr) > 8000:  # 4000: #self.max_steps: #or
        #         # print('I became an old man and dies in this maze...')
        #         return self.observe_environment(), -0., True, {}  # terminate
        #
        #     '''Handle Invalid Movment'''
        #     if not valid_move:
        #         self.meta_environment.update_history(self.actor_pos)  # update meta-environment
        #         return self.observe_environment(), -0, False, {}
        #
        #     '''Valid Movement'''
        #     self.actor_pos = new_pos  # set new position
        #     self.meta_environment.update_history(new_pos)  # update meta-environment
        #
        #     '''If we are re-tracing steps'''
        #     if new_pos in self.actorpath:
        #         self.moved_max = len(self.actorpath) - 1
        #         score = self.prior_scores[-1]
        #         self.prior_scores.pop(-1)
        #         return self.observe_environment(), -score, False, {}
        #
        #     # Have we reached the end?
        #     if self.actor_pos == (199, 199):
        #         print('Final achieved')
        #         return self.observe_environment(), 10000., True, {}
        #
        #     velocity = self.movement_velocity(new_position=new_pos)
        #
        #     score = 0
        #     if new_pos in optimal_path:
        #         score = (1 - (len(self.actorpath) / 3600))  # / self.step_since_move #* velocity
        #
        #     self.prior_scores.append(score)
        #
        #     self.step_since_move = 0  # reset
        #     return self.observe_environment(velocity), score, False, {}
