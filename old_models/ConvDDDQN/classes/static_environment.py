"""Environment Class will manage the observation and reward functions

Notes
-----
The Environment class contains all information and methods surrounding observation
For simplicity and ease of optimisation we use `action_dir` and `reward_dir` to describe the action and reward
spaces.




"""
from old_models.DDQN.lib.read_maze import get_local_maze_information

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
rewards_dir = {"towards": +1,
               "away":+1,
              "visited":-1.,
              "wall":-1.,
              "stay":-1.
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

        if self.actor_pos not in self.actorpath:
            self.visit_cntr = 0
            self.actorpath.append(self.actor_pos)
        else:
            self.visit_cntr += 1


        self.observation = l1 + l2 + [self.actor_pos[0], self.actor_pos[1]] # , self.step_cntr ]# + [prior_pos[0], prior_pos[1]]
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
        if self.step_cntr > len(self.actorpath)*2:
            print('I became an old man and dies in this maze...')
            return self.observe_environment, -10., True, {} # terminate
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
