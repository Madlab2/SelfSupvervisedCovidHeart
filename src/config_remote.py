from utils import *

### Data Config ###
DATA_PATH = convert_path('/dtu/3d-imaging-center/courses/02510/data/CovidHeart/covid_small')

WINDOW_OVERLAP = 0.5
PATCH_SIZE= (96,) * 3         # Size of crops
PROB_FOREGROUND_CENTER= 0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

### Pretraining Config ###
PRE_NUM_EPOCHS = 10
PRE_TRAIN_BATCH_SIZE = 5
PRE_BATCHES_PER_EPOCHS = 100
PRE_LR = 1e-4

PRE_VAL_BATCH_SIZE=5

### Training Config ###
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 5
BATCHES_PER_EPOCHS = 250
LR = 1e-4
DROPOUT = 0.2                # Should be turned off during validation/inference?

### Validation Config ###
VAL_BATCH_SIZE=5

### Inference Config ###
INFERENCE_BATCH_SIZE = 2