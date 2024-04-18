import os

### Data Config ###
DATA_PATH='.\data\CovidHeart\covid_small'
if os.name == 'nt':
    # Windows Path
    DATA_PATH = DATA_PATH.replace('/', '\\')


WINDOW_OVERLAP = 0.5
PATCH_SIZE= (96,) * 3         # Size of crops
PROB_FOREGROUND_CENTER= 0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

### Training Config ###
TRAIN_BATCH_SIZE = 16
BATCHES_PER_EPOCHS = 150
NUM_EPOCHS = 5
LR = 1e-4
DROPOUT = 0.2                # Should be turned off during validation/inference?

### Validation Config ###
VAL_BATCH_SIZE=16

### Inference Config ###
INFERENCE_BATCH_SIZE = 16