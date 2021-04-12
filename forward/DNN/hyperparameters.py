"""
Parameter file for specifying the running parameters for forward model
"""
# Model Architectural Parameters
USE_CONV = True
LINEAR = [4,750,750,750,750,750,750,250]
CONV_OUT_CHANNEL = [8, 4, 4]
CONV_KERNEL_SIZE = [16, 5, 5]
CONV_STRIDE = [2, 1, 1]

# Optimization parameters
OPTIM = "Adam"
REG_SCALE = 1e-4
BATCH_SIZE = 1024
EVAL_STEP = 10
RECORD_STEP = 50
TRAIN_STEP = 500
LEARN_RATE = 1e-3
# DECAY_STEP = 25000 # This is for step decay, however we are using dynamic decaying
LR_DECAY_RATE = 0.5
STOP_THRESHOLD = 1e-5
USE_CLIP = False
GRAD_CLIP = 50
USE_WARM_RESTART = False
LR_WARM_RESTART = 400

# Data Specific parameters
X_RANGE = [i for i in range(0, 4)]
Y_RANGE = [i for i in range(1, 1001,2)]
FREQ_LOW = 20.02
FREQ_HIGH = 40
NUM_SPEC_POINTS = 500
FORCE_RUN = True
# DATA_DIR = ''                # For local usage
DATA_DIR = 'C:/Users/labuser/DL_AEM/'                # For Omar office desktop usage
# DATA_DIR = 'C:/Users/Omar/PycharmProjects/DL_AEM/' # For Omar home desktop usage
# DATA_DIR = '/home/omar/PycharmProjects/DL_AEM'  # For Omar laptop usage
# Format for geoboundary is [p0_min... pf_min p0_max... pf_max]
GEOBOUNDARY =[1.3, 0.975, 6, 34.539, 2.4, 3, 7, 43.749]
NORMALIZE_INPUT = True
TEST_RATIO = 0.2
DATA_REDUCE = 0

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None
EVAL_MODEL = "20k_Dataset"
NUM_PLOT_COMPARE = 10
