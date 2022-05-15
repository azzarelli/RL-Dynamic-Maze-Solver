"""Providing Prioritised Experience Replay and Random Experience Replay Class-Methods

"""
import random
import numpy as np
from ConvDDQN.classes.sumtree import SumTree

'''PER Class Definition
        Note that PER does not support input of multiple frames (this is only a featue the Random ER class can handle)
'''
class PrioritizedBuffer:
    def __init__(self, max_size, batch_size, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size) # Using SumTree will reduce computation necessary for tracking memory
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.current_length = 0 # curent length of memory

    def add(self, state, next_state, reward,  action, done):
        """Adding experience to memory
        :param state, next_state: Current and following states
        :param reward: Reward
        :param action: Action taken by agent
        :param done: Bool deifining termination of game at current state

        Notes
        ----
        Here we access our SumTree (which keeps track of memory using prioritisation hierachical tree) to define
        priorities relative to most recent step. Priorities of inputs will be changed while learning is accomplished,
        refer to `update_priorities` method.
        """
        priority = 1.0 if self.current_length == 0 else self.sum_tree.tree.max()
        self.current_length = self.current_length + 1

        img = state.numpy() # Convert Tensor to numpy to save experience
        state = img
        img_ = next_state.numpy()
        next_state = img_
        experience = (state, action, np.array([reward]), next_state, done)
        self.sum_tree.add(priority, experience) # add experience and initial priority to SumTree

    def sample(self):
        """Sample experience dependant on prioritisation of states (`alpha`) and prioritisation of fetched memory (`beta`)

        Notes
        -----
        We  firstly segment our sumtree (saved expeirences) into equal chunks of equal to batchsize. Within in each
        seqment we fetch priorities of randomly selected data and determined its sampling weight in output batch.
        We then re-format the outputs to cohere with expected inputs for learning method in `agent.py`
        """
        batch_idx, batch, IS_weights = [], [], []
        segment = self.sum_tree.total() / self.batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b) # randomly sample a index of memory to fetch which exists within the segment
            idx, p, data = self.sum_tree.get(s) # fetch position (relative to sumtree NOT segement), priority and experience

            batch_idx.append(idx) # track positions of experience in output batch
            batch.append(data) # append experience
            prob = p / p_sum # determine probaility of being choisen dependant on priority relative to sum priorities
            IS_weight = (self.sum_tree.total() * prob) ** (-self.beta) # Imporatnce sampling weight
                                                                        #   w_i = (1/N * 1/P(i))^beta = (N*P(i))^-beta
            IS_weights.append(IS_weight) # track weight

        '''Modify output of PER to expected output of learning method (refer to `agent.py`)'''
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []
        for transition in batch:
            state, action, reward, next_state, done = transition
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return np.array(state_batch), action_batch, np.array(reward_batch), np.array(next_state_batch), done_batch, IS_weights, batch_idx


    def update_priorities(self, idx, td_error):
        """Update the priorities of each experience within the SumTree (after one learning step)
        :param idx: batch of indexs of relative experience
        :param td_error: TDErrors calulcated after learn-step to determine new priority of experiences
        :return:
        """
        priority = td_error ** self.alpha
        self.sum_tree.update(idx, priority) # update priorities in sumtree

    def is_sufficient(self):
        """Check memory has been filled before sampling (if not prevent learning untill experience is gathered)
        """
        return self.current_length > self.batch_size

'''Random ER Class Definition'''
class RandomBuffer():
    def __init__(self, input_shape, max_size, batch_size, multi_frame:bool=True):
        self.multi_frame = multi_frame # choice of saving multiple frames
        self.batch_size = batch_size
        self.mem_size = max_size # bound memory so we don't crash RAM
        self.mem_cntr = 0 # simulate stack by knowing whee in memory we are

        input_shape = (input_shape[0], input_shape[1], input_shape[2]) # reduncancy check

        '''Initialise Memory'''
        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.state_mem_ = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_mem = np.zeros(self.mem_size, dtype=bool)

    def add(self, state, state_, reward,  action, done):
        """Add experience to memory

        :param state, state_: Current and future states
        :param reward: Reward
        :param action: Action
        :param done: Termination flag

        Notes
        -----
        Unlike PER we can directly commit experience to memory as fetching will be randomised
        """
        # wrap counter to know where we are in memory
        idx = self.mem_cntr % self.mem_size

        if self.multi_frame == True:
            # state = transforms.ToPILImage()(state)
            state = state.numpy()
            state_ = state_.numpy()

        # assign states, reward and action to memory
        self.state_mem[idx] = state
        self.state_mem_[idx] = state_
        self.reward_mem[idx] = reward
        self.action_mem[idx] = action
        self.terminal_mem[idx] = done
        # increment position in memory
        self.mem_cntr += 1

    def sample(self):
        """Random sampling of batch in memory
        """
        # Max existing memory size (if mem not full set max mem to counter value)
        max_mem = min(self.mem_size, self.mem_cntr)
        btch = np.random.choice(max_mem, self.batch_size, replace=False) # fetch non-repeating experience from memory

        states = self.state_mem[btch]
        states_ = self.state_mem_[btch]
        actions = self.action_mem[btch]
        rewards = self.reward_mem[btch]
        terminal = self.terminal_mem[btch]

        return np.array(states), actions, np.array(rewards), np.array(states_), terminal

    def is_sufficient(self):
        """Check memory has been filled before sampling (if not prevent learning untill experience is gathered)
        """
        return self.mem_cntr > self.batch_size