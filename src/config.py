import os

INFERENCE_BATCH_SIZE = 16
WINDOW_OVERLAP = 0.5

DATA_PATH='.\data\CovidHeart\covid_small'
if os.name == 'nt':
    DATA_PATH = DATA_PATH.replace('/', '\\')

PATCH_SIZE=(96,) * 3         # Size of crops
PROB_FOREGROUND_CENTER=0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)
TRAIN_BATCH_SIZE=16
BATCHES_PER_EPOCHS=150
VAL_BATCH_SIZE=16

LR = 1e-4
NUM_EPOCHS = 5
DROPOUT = 0.2