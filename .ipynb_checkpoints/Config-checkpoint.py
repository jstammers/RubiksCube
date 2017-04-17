# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = False
# Gym environment
ENV_NAME = 'Rubiks'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 500
#Size of Rubiks Cube
N_CUBE=2

# ===========================
#   Recurrent Network Parameters
# ===========================
#Fraction of data used for training
TRAIN_FRAC = 0.7
#Flags Generation of Data
GENERATE_DATA = True
# Base Learning Rate
LEARNING_RATE = 0.001
# Training Epochs
EPOCHS = 5
# Number of steps to test if it can solve a cube
TEST_STEPS = 50
