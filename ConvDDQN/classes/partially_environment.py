"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir`  to describe the action space. `digits_dir` provides the
bit-maps for the digits if we choose to provide numerical values over our input image.
We have also included the class method defining the meta-state environment which provides us with the traffic of the actor over
the learning period


"""

from ConvDDQN.lib.read_maze import get_local_maze_information, quick_get_local_maze_information
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.transforms import transforms
from torchvision.utils import save_image

import torch as T

global digit_dir
digit_dir = {'none':[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]],
             '0':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,1,0,1,0,0],[0,1,0,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '1':[[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,1,0,0,0],[0,0,0,0,0,0]],
             '2':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,1,1,1,0,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '3':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '4':[[0,0,0,0,0,0],[0,1,0,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]],
             '5':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '6':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '7':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]],
             '8':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,0,0,0,0,0]],
             '9':[[0,0,0,0,0,0],[0,1,1,1,0,0],[0,1,0,1,0,0],[0,1,1,1,0,0],[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0]]}

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
    def __init__(self, img_size:int=10):
        """Set the primary attributes of the environment handler

        Attributes
        ----------
        window_size: int, size of the original image frame (this will be enlarged to 40x40 image size as defined in `run.py`)
        visible_size: int, the radial distance of the square oberservation matrix (=1 means 3x3 pixel observation, =2 means 5x5, etc.)
        """
        self.window_size = img_size
        self.visible_size = 1

        self.reset() # set the secondary attributes (these are class attributes which are reset every episode)

        self.meta_environment = MetaEnvironment() # initialise the meta-state which tracks the frequency of steps for each position on the maze

    def reset(self, start_position:tuple=(1,1), max_steps:int=3500, block:tuple=(0,0)):
        """Reset the environment at the beginning of every episode

        Attributes
        ----------
        stay: int, indicator of actor staying
        step_cnts, wall_cntr, stay_cntr, fire_cntr, visit_cntr: int, these are counters for tracking environment behaviours
        step_since_move: int, denotes the steps since the last valid move (used for positive reward function)
        score: float, current score of the environment
        valid_move: bool, denotes where an action taking is valid
        prior_scores: list of float, saves the scores of the environment alongisde each step

        start_position: tuple, the x and y coordinates we want the actor to start from
        block: tuple, x and y values of a path we want to block (this isnt used but could be helpful for warm-up strategies)
        direction: string, denotes the direction our actor is moving (to be called externally)

        actos_pos: tuple, x and y position of the actor
        actor_path: list of tuples, denoting the position of the path the has passed
        observation_map: list of list of RGB values, denotes the gloab map of the maze
        img_history: list of matrices, denotes the list of global observation made though an epsidoe
        pos_history: denotes the positions which resulted in images in image history (indexs are relative)
        hist_idx: int, denotes the position in img_history we want to revisit
        last_hist_idx: int, denotes the last position in img_history we contributed

        obs2d: 3x3 matrix, denotes the local observation used for drawing pygame canvas
        observation: Tensor, contains the local observation which we use as out input
        """
        self.stay = 0
        self.step_cntr = 0
        self.step_since_move = 0
        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0
        self.fire_cntr = 0
        self.score = 0
        self.valid_move = True
        self.prior_scores = [1.]

        self.start_position = start_position
        self.block = block

        self.direction = 'stay'

        self.actor_pos = start_position
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

    """Function for transforming a PIL image into a tensor"""
    transform = transforms.Compose([transforms.ToTensor()])

    def get_split_step(self):
        """Function calling the counters which cound invalid moves - for canvas"""
        return [self.stay_cntr, self.visit_cntr, self.wall_cntr]

    def render_global_map(self, loc):
        """Update the global map for tracking observations made all over the maze

        :param loc, 9x9 matrix, defines the local observation
        """
        x, y = self.actor_pos # fetch the actos position to fetch the local observation
        pos = [(x + j - 1, y + i - 1) for i in range(3) for j in range(3)]

        '''Update the global map with the local observation'''
        for a, b in pos:
            i, j = a - x + 1, b - y + 1  # (0,0) (0,1)(0,2), (1,0) ... (2,2)
            if loc[j][i][0] == 0:  # wall
                self.observation_map[b][a] = WALL
            elif loc[j][i][1] > 0:  # fire
                self.observation_map[b][a] = FIRE
            else:  # path
                self.observation_map[b][a] = EMPTYPATH
            if (a,b) == self.block:
                self.observation_map[b][a] = WALL
                self.loc[j][i][0] = 0

        # Convert actor path to colour of global map
        for a, b in self.actorpath:
            self.observation_map[b][a] = PATH

        '''Draw in the actors position and '''
        self.observation_map[y][x] = ACTOR
        self.observation_map[199][199] = END


    def get_digits(self, obs, getType):
        """Fetch digits to paste into image array

        Notes
        -----
        We call the digit_dir list constaining the bit-map for digits and synthesise and new image with additional
        numerical indicators (specifically we have indicated the steps since last moving)

        """
        if getType == '4digit_step_move':
            '''Define the numerical value to visualise'''
            path_len = len(self.actorpath) # can define any integer values between 0 and 9999

            '''Find the digit representations'''
            p_digit_num = (int(path_len / 1000), int(path_len / 100) % 10, int(path_len / 10) % 10, path_len % 10)

            '''Fetch the relevant digits from digit_dir'''
            p_digits = []
            for p_i in p_digit_num:
                p_d = np.array(digit_dir[str(p_i)])
                p_digits.append(p_d)
            p_digits = np.concatenate(p_digits, axis=1)

            '''Past the binary digit maps onto the image'''
            a, b = 10,2 # these are the pixel values for number position (representing the top-left corener of digits pixel map)
            for j, pos_y in enumerate(p_digits):
                for i, pos in enumerate(pos_y):
                    if pos == 1: # Past 1. (white) and 0. (black) relative to bitmap
                        obs[0][b + j-1][a + i-1] = T.tensor(1.)
                        obs[1][b + j - 1][a + i - 1] = T.tensor(1.)
                        obs[2][b + j - 1][a + i - 1] = T.tensor(1.)
        return obs



    def render_local_state(self):
        """Render the local state, which will be be processed for input into out DQN

        Notes
        -----
        To populate the image we take a snapshot of of the environment we have explored in the episode and reduce the
        window size so that we see a sub-map of the global map.
        """
        x, y = self.actor_pos # fetch the actors position

        '''Calculate range of visible to the agent in the local map
                (combines local observation with learnt environment)
        '''
        diffx = 5 #int((self.window_size) / 2)
        diffy = 5
        x_lb, x_ub = x - diffx, x + diffx
        y_lb, y_ub = y - diffy, y + diffy

        obs = [[[0, 0, 0] for i in range(self.window_size)] for j in range(self.window_size)] # initilise observation matrix
        for j, y_i in enumerate(range(y_lb, y_ub+1)):
            for i, x_i in enumerate(range(x_lb, x_ub+1)):
                v = self.visible_size # the perpendicular distance visible to our DQN agent

                # Observations which exist outside of the global map are processed, otherwise ignored
                if (x_i >= x-v and x+v >= x_i) and (y_i >= y-v and y+v >= y_i):
                    if y_i < 0 or x_i < 0 or y_i > 199 or x_i > 199:
                        obs[j][i] = WALL
                    else:
                        '''Append known pizel-states to enlarged local observation matrix'''
                        obs[j][i] = self.observation_map[y_i][x_i]


        return obs

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
        loc = quick_get_local_maze_information(y, x) # fetch the local observation at the lowest-level
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

        '''Now the global state space has been updated, we can update the actor's path'''
        if self.actor_pos not in self.actorpath and self.stay == 0:
            self.actorpath.append(self.actor_pos)
        self.stay = 0

        '''Fetch RGB and Greyscale representations of local enlarged local statespace
                Notes - Enlarging visual region from 3x3 to nxn where n>3, means we don't run into seemingly repeated 
                        observtions for 3x3 blocks which are same but on different parts of the mase, with more info of 
                        environment states become more unique 
        '''
        obs = self.render_local_state()

        '''Image Processing for numpy array into suitable tensor'''
        obsv_ = np.array(obs, dtype=np.uint8)
        img = Image.fromarray(obsv_, 'RGB')
        img = img.resize((40, 40), resample=Image.NEAREST)

        imgs = self.transform(img)
        imgs = self.get_digits(imgs, '4digit_step_move')

        self.obs2D = np.array(img).copy() # save enlarged observations for other processes
        self.obs = img

        # Choice to save images
        show_img = 0
        if show_img:
            save_image(imgs, 'obs_col.png') # save coloured obseervation
        show_channels = 0
        if show_channels:
            save_image(imgs[0], 'img_R.png') # track each channel
            save_image(imgs[1], 'img_G.png')
            save_image(imgs[2], 'img_B.png')

        return imgs

    @property
    def get_actor_pos(self):
        return self.actor_pos

    def valid_movement(self, next_local_position):
        """Return the validity of move and reason for valid/invalidity
        """
        valid_move = True # initialise a move as valid until proven guilty
        reason = ''
        x_new, y_new = next_local_position
        obsv_mat = self.loc  # get prior position

        '''Check road-blocks (fire & wall) - For dead ends & fire wall
        '''
        paths = 0
        fire_block = 0
        for i, o in enumerate(obsv_mat):
            for j, s in enumerate(o):
                if [i, j] in [[0, 1], [1, 0], [2, 1], [1, 2]]:
                    if s[0] != 0:
                        paths += 1
                    if s[1] > 0:
                        fire_block += 1
        fire_block = 0 # delet for dynamic fire
        '''Staying Put'''
        if next_local_position == (1,1):
            if fire_block > 0: # we should stay if fire is near us
                valid_move = True
                reason = 'fireblock'
            else: # otherwise we penalise staying
                valid_move = False
                reason = 'stay'
        else:
            # Dead end
            if paths == 1:
                valid_move = False
                reason = 'deadend'
            # Wall Blocking
            elif obsv_mat[y_new][x_new][0] == 0:
                valid_move = False
                reason = 'wall'
            # Fire Blocking
            elif obsv_mat[y_new][x_new][1] > 0:
                valid_move = False
                reason = 'fire'

        self.valid_move = valid_move
        return valid_move, reason

    def step(self, action, score):
        """Sample environment dependant on action and return the relevant reward
        """
        self.score = score # save score to

        '''Increment counters and initialise environment variable'''
        self.step_cntr += 1
        self.step_since_move += 1
        self.valid_move = True # initilised the valid move flag
        global action_dir # Fetch action directory containing the properties of each action w.r.t environment
        act_key = str(action) # note action as a string {'up','left','right','down','stay'}

        '''Determine the expected movement (increment/decrement values for x,y positions)'''
        x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)
        self.direction = action_dir[act_key]['id'] # save the direction of movement
        x,y, = self.actor_pos # fetch the actors global position
        new_pos =  (x + x_inc, y + y_inc) # determine expects new position (we will validate this move later)
        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position


        '''Validate the movement and update movement counters'''
        valid_move, reason = self.valid_movement((x_loc, y_loc))
        if not valid_move: # update wall, fire and stay counters
            self.update_counters(reason)
        elif new_pos in self.actorpath: # otherwise update re-visit counter
            self.visit_cntr += 1

        '''Termination for deadend or duration:'''
        if reason == 'deadend':
            #print('deadend')
            return self.observe_environment(), -10., True, {}  # terminate
        if self.step_since_move == 50:
            #print('step death...')
            return self.observe_environment(), -10., True, {} # terminate
        if abs(self.step_cntr) > 8000:
            #print('I became an old man and dies in this maze...')
            return self.observe_environment(), -0., True, {} # terminate

        '''Handle (other) Invalid Movments'''
        if not valid_move:
            self.meta_environment.update_history(self.actor_pos) # update meta-environment
            return self.observe_environment(), -0., False, {}

        '''When fire blocks our paths, reward patience'''
        if reason == 'fireblock':
            self.step_since_move = 1 # reset the step since target movement counter
            self.meta_environment.update_history(self.actor_pos) # update meta-environment
            return self.observe_environment(), 1., False, {}  # reward staying if fire is blocking

        '''Valid Movement'''
        self.actor_pos = new_pos # set new position
        self.meta_environment.update_history(new_pos) # update meta-environment

        '''If we are re-tracing steps'''
        if new_pos in self.actorpath:
            self.moved_max = len(self.actorpath)-1
            self.prior_scores.pop(-1)
            self.step_since_move = 0
            return self.observe_environment(), -1., False, {} # negativly penalise retracing steps

        # Have we reached the end?
        if self.actor_pos == (199, 199):
            print('Final achieved')
            return self.observe_environment(), 10000., True, {}

        '''Determine the score depedant on number of steps since valid movement'''
        score = 1/self.step_since_move if 1/self.step_since_move > 0.1 else 0.1  # (1 - (len(self.actorpath) / 3600)) #*0.01 #/ self.step_since_move #* velocity

        self.prior_scores.append(score) # append score
        self.step_since_move = 0 # reset movement counter
        return self.observe_environment(), score, False, {}

    def update_counters(self, reason):
        self.stay = 1
        if reason == 'wall':
            self.wall_cntr += 1
        elif reason == 'stay':
            self.stay_cntr += 1
        elif reason == 'fire':
            self.fire_cntr += 1


class MetaEnvironment:
    """Class method which illustrates the frequency of movement around the maze by counting the number of steps an actor
    has made in each position. 'Hot' regions will denote areas of high traffic, hence confirmation of learnt positions
    """
    def __init__(self):
        self.environment_history = np.zeros((200,200)) # maze position counter

    def update_history(self, global_position):
        """Update the heat map"""
        x,y = global_position
        self.environment_history[y][x] += 1

    def save_meta_experience(self, fp:str='META_experience.data'):
        """Save the map as a heat-map"""
        hist = self.environment_history.copy()
        hist[hist==0] = np.NAN # Turn all unexplored elements to white (easier to visualise)
        plt.imshow(hist, cmap='hot', interpolation='nearest')
        plt.savefig('liveplot/METADATA.png')
