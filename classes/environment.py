import time
from lib.read_maze import get_local_maze_information

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

global reward_dir
reward_dir = {"stay": 0,
              "wall":-1,
              "fire":-50,
              "oldman":-50,
              "away":0,
              "back":0,
              "end":10000
              }


class Environment():

    def __init__(self):
        self.step_cntr = 0
        self.actor_pos = (1, 1)
        self.observation = self.observe_environment  # set up empty state

    @property
    def reset(self):
        self.step_cntr = 0
        self.actor_pos = (1, 1)
        self.observation = self.observe_environment  # set up empty state
        return self.observation

    @property
    def observe_environment(self):
        x,y = self.actor_pos
        loc = get_local_maze_information(y,x)
        l1, l2 = [], []
        for l in loc:
            for j in l:
                l1.append(j[0]) # index % 2 in loc_vec = wall data
                l2.append(j[1]) # index % 3 (and index 1) in loc_vec = fire data
        self.observation = l1+l2
        #self.observation.append(self.step_cntr) # time elapsed playing is also an input
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
        loc_mat = [[[0, 0] for i in range(3)] for j in range(3)]
        for i in range(len(self.observation)):
            if i < 9:
                loc_mat[int(i/3)][i - int(i/3)*3][0] = self.observation[i]
            else:
                j = i - 9
                loc_mat[int(j / 3)][j - int(j / 3) * 3][1] = self.observation[i]
        return loc_mat

    @property
    def get_actor_pos(self):
        return self.actor_pos

    def step(self, action):
        """Sample environment dependant on action which has occurred

        Action Space
        ------------
        0 - no move
        1 - up
        2 - left
        3 - down
        4 - right

        Environment Rules
        -----------------
        - if we try to walk into a wall our character stays fixed, v small penalty for simply not choosing to stay
        - if we walk into a fire, the game is terminated and we give a penalty
        - if we take a step away fom actors prior position (maybe ref 1,1, or actual prior pos) reward,
        - however if we take a step back from end point, reward = 0
        - if we reach 199*199 we receive a reward of `R` (dependant on the number of steps it took to get there)

        - TODO - Later we could give rewards for not moving when all paths are blocked by fires
        """
        #time.sleep(0.05) # delay for time animation
        self.step_cntr += 1 # increment time

        # Fetch action directory containing the properties of each action w.r.t environment
        global action_dir
        act_key = str(action)
        # Fetch reward directory
        global reward_dir

        # fetch movement from position (1,1)
        x_inc, y_inc = action_dir[act_key]['move']

        # If too much time elapsed you die in maze :( (terminate maze at this point)
        if self.step_cntr > 1200:
            print('I became an old man and dies in this maze...')

            return self.observe_environment, reward_dir['oldman'], True, {}

        # Check if we have moved
        if action_dir[act_key]['id'] == 'stay':
            return self.observe_environment, reward_dir['stay'], False, {}

        x, y = self.actor_pos

        # Update Local Position
        x_loc, y_loc = new_loc = (1 + x_inc, 1 + y_inc)

        # Check if Local Position viable
        obsv_mat = self.get_local_matrix

        # Get available routes (if the exist)
        routes = 1 # we always have a choice to 'stay', thus possile routes to avoid death >= 1
        for i in range(1, 5):
            x_, y_ = action_dir[str(i)]['move']
            ob0 = obsv_mat[y_][x_][0]
            ob1 = obsv_mat[y_][x_][1]
            # if no fire or wall save path (for criticising choices later on)
            if ob0 == 1 and ob1 == 0:
                routes += 1

        #print(f'Obs: {obsv_mat} \n {obsv_mat[y_loc][x_loc]}')
        # Check existence of wall
        if obsv_mat[y_loc][x_loc][0] == 0:

            rew_ = reward_dir['wall'] * routes # if there are routes penalise the stupidity of walking into walls
            return self.observe_environment, rew_, False, {}

        # Check if we have jumped into fire
        if obsv_mat[y_loc][x_loc][1] > 0:
            rew_ = reward_dir['fire'] + (10* routes) # amplify penalisation if many routes existed
            # reward = reward_dir['fire'] + max([x,y]) # incentive to die further away
            return self.observation, rew_, True, {}


        #print('Successful move!')
        self.actor_pos = new_pos = (x + x_inc, y + y_inc) # new global position

        # Have we reached the end?
        if new_pos == (199, 199):
            return self.observation, reward_dir['end'], True, {}

        rew = reward_dir['away'] if x_inc > 0 or y_inc > 0 else reward_dir['back']
        return self.observe_environment, rew, False, {}