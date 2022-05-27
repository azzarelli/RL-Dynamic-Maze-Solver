# Dynamic-Maze-Solver (DQN/DDQN)
####COMP6247 - Solving a Large Dynamic Maze Problem

We use convolutional DQNs and DDQNs (with LSTM flavour) to structure an agent
to solve a dynamically changing maze (i.e.) random apartitional fires which dies out after n < 3
time-steps.

We also have handlers for the environment processing (encapsulating image processing, state-space tracking, reward function) 
and actor network (for learning, exploration memory control).

## Compiling and Running
We have provided a `requirements.txt` file containing the package requirements to run this project.


To run the program simply open up the `main.py` file and configure the setting you wish to use. Current
setting exist for **training** (REMEBER to set a new `name` for different runs). 

If you want to test an pre-trained network, change: `canv_chck=1,  train_chck=True, chckpt=True, epsilon=0.0`, other 
parameters will be ignored/remain the same. The testing function will also provide the output file
as a *json* providing the sequence of paths, local observation, actions, steps position.

## File-Directory Structure: 
    ConvDDQN/
    ├── classes                           # Algorithm & Plotting Classes/Handlers
    |      ├── agent.py                  # Agent/Actor Class (network learning
    |      |                                                   /action selection, etc.) 
    |      ├── canvas.py                 # Visualise Global Environment (w/pygame)
    |      ├── convddqn_lstm.py          # DDQN w/LSTM network class
    |      ├── convdqn_lstm.py           # DQN w/LSTM network class
    |      ├── partially_environment.py  # Handles active environment and state  
    |      |                                                           image processing 
    |      ├── plotter.py                # Handller for monitoring training handler
    |      ├── replaybuffer.py           # Uniform sampling Exp. Replay and PER classes
    |      ├── sumtree.py                # Handles Memory for PER using Sumtree (efficient)
    |      └── training.py               # Training method 
    ├── lib                    # Provided files (`load_maze`, `get_local_maze_information`
    |                                                & `quick_get_local_maze_information`)
    ├── liveplot               # Folder containing live graphs for running training
    ├── networkdata            # Folder for saved networks
    ├── main.py                # Main execution file (tunable parameters) (calls `run.py`)
    ├── run.py                 # Run file containing fixed parameters (e.g.) 
    |                           network type:{DDQN, DQN}, Experience Replay:{Random, Priority}...
    └── README.md

**Notes:** `main.py` is our main executable where external tunable parameters like batchsize, learning rate, etc. 
can be set. `run.py` is executed by main and contains architectural paramets like network type 
input dimensions (size of image), environment & agent initialisation, ... . `training.py` is the consequence
of this and runs the training method for our algorithm - within here we run transitions and
learning steps (as well as tracking training data). We also added a function to `lib/read_maze.py` which accelerated
the very expensive method for feeding local-observations into our algorithm - this doesn't
comprimise the integrity of the program and can be reverted if desied.

## Other Files/Folders
`old_models/...` contains (undocumented) depricated versions of our program, such as
non-convolutional DQN and DDQN as well as Rainbow DQN and Double DQN networks. These have not
been tuned (and likely contain old versions of the environment). They are usable - though we suggest making
necessary modifications where you can. (The sub-folders are similarly structured to the structure presented
above).

`Manual_Path_Finder/...` is a version of the game we can use manually (no Agent) and has been used 
to solve the game to provide information on the optimal (static) solution. This has not been used in training
our agent and is simply for visual purpose. (Similar folder-organisation to structure above).

