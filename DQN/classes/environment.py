"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
from DDQN.lib.read_maze import get_local_maze_information

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
rewards_dir = {"onwards": -.9,
              "backwards":-.9,
              "wall":-2.,
              "stay":-1.,
              "visited":-1.,
              "deadend":-10.
              }


class Environment:
    def __init__(self):
        self.step_cntr = 0
        self.actor_pos = (1, 1)
        self.actorpath = [self.actor_pos]
        self.observation = self.observe_environment  # set up empty state
        self.obs2D = []

        self.wall_cntr = 0
        self.stay_cntr = 0
        self.visit_cntr = 0



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
                l1.append(loc[l][j][0])

                x_ = x + j - 1
                y_ = y + l - 1

                # if (x_, y_) in self.actorpath:
                #     l2.append(1)
                # else:
                #     l2.append(0)
                #     self.actorpath.append(self.actor_pos)


        self.observation = l1 + [self.actor_pos[0], self.actor_pos[1]] + [self.actorpath[-1][0], self.actorpath[-1][1]]
        if (x,y) not in self.actorpath:
            self.actorpath.append((x,y))
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
        #time.sleep(5) # delay for time animation
        self.step_cntr += 1 # increment time

        global action_dir # Fetch action directory containing the properties of each action w.r.t environment
        act_key = str(action)
        global rewards_dir # Fetch reward directory

        x_inc, y_inc = action_dir[act_key]['move'] # fetch movement from position (1,1)

        # If too much time elapsed you die in maze :( (terminate maze at this point)
        if self.step_cntr > 500:
            print('I became an old man and dies in this maze...')
            return self.observe_environment, -1., True, {} # terminate

        obsv_mat = self.get_local_matrix # get prior position
        x, y = self.actor_pos
        x_ = x + x_inc
        y_ = y + y_inc

        x_loc, y_loc = (1 + x_inc, 1 + y_inc) # Update Local Position


        if action_dir[act_key]['id'] == 'stay': # if we stay for no reason then penalise
            self.stay_cntr += 1
            return self.observe_environment, rewards_dir['stay'], False, {}

        if obsv_mat[y_loc][x_loc][0] == 0: # check for a wall
            self.wall_cntr += 1
            return self.observe_environment, rewards_dir['wall'], False, {}

        self.actor_pos = new_pos = (x + x_inc, y + y_inc) # new global position if we move into a free space

        # Check to see if we are either blocked in as a result of prior path, wall or fire (if so -> reward staying)
        is_blocked = True
        for i, o in enumerate(obsv_mat):
            for j, p in enumerate(o):
                if (i,j) in [(0,1), (1,0), (1,2), (2,1)]:
                    pos = (x + j - 1, y + i -1)
                    if p[0] == 1 or (pos not in self.actorpath):
                        is_blocked = False
        if is_blocked : # Reward staying if path is blocked
            print('Deadend')
            return self.observe_environment, rewards_dir['deadend'], True, {}


        # Have we reached the end?
        if new_pos == (199, 199):
            return self.observation, 100., True, {}

        if (x_, y_) == self.actorpath[-1]:
            return self.observe_environment, rewards_dir['visited'], False, {}

        # Are we moving towars the goal?
        if x_inc > 0 or y_inc > 0:
            return self.observe_environment, rewards_dir['onwards']*float(len(self.actorpath)**-1), False, {}

        # finally our only choice is to move away from goal
        return self.observe_environment, rewards_dir['backwards']*float(-len(self.actorpath)**-1), False, {}
