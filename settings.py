import os

# Environment parameter
VERBOSE = True
DEBUG 	= True
VISUALISE = False
DEBUG_SIZE = 2

# Model Hyperparameter
TRAIN_SIZE = 3696
VAL_SIZE = 184
TEST_SIZE = 1344

# Path
DATA_PATH = os.path.join(os.getcwd(), 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'TRAIN')
TEST_PATH = os.path.join(DATA_PATH, 'TEST')
TARGET_PATH = os.path.join(DATA_PATH, 'target_preprocessed_data.pkl')

