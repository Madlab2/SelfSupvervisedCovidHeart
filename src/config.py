import sys
from utils import *

if sys.path[0].find("dtu/3d-imaging-center") != -1:
    # remote

    ### Data Config ###
    DATA_PATH = convert_path('/dtu/3d-imaging-center/courses/02510/data/CovidHeart/covid_small')
    SCRATCH_MODEL = 'models/worst_model_checkpoint.pth'   # no pretraining
    PRE_MODEL_NAME = 'models/pretrain/real_scratch/pretrain_model_checkpoint_e37_loss0.004473115783184767.pth' # specifiy path for pretrained model

    PRE_TRAIN_NOISVAR = 0.01
    WINDOW_OVERLAP = 0.5
    PATCH_SIZE= (96,) * 3         # Size of crops
    PROB_FOREGROUND_CENTER= 0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

    ### Pretraining Config ###
    PRE_NUM_EPOCHS = 40
    PRE_TRAIN_BATCH_SIZE = 4
    PRE_BATCHES_PER_TRAIN_EPOCH = 25
    PRE_LR = 1e-4

    PRE_BATCHES_PER_VAL_EPOCH = 10
    PRE_VAL_BATCH_SIZE=4

    ### Training Config ###
    NUM_EPOCHS = 100
    TRAIN_BATCH_SIZE = 32
    BATCHES_PER_EPOCHS = 50
    LR = 1e-4
    DROPOUT = 0.2                # Should be turned off during validation/inference?

    ### Validation Config ###
    VAL_BATCH_SIZE=32

    ### Inference Config ###
    INFERENCE_BATCH_SIZE = 32

else:
    # local

    ### Data Config ###
    DATA_PATH = convert_path('./data/CovidHeart/covid_small')
    SCRATCH_MODEL = './models/worst_model_checkpoint.pth'   # no pretraining
    PRE_MODEL_NAME = 'model_checkpoint.pth'
    
    PRE_TRAIN_NOISVAR = 0.01
    
    WINDOW_OVERLAP = 0.5
    PATCH_SIZE= (96,) * 3         # Size of crops
    PROB_FOREGROUND_CENTER= 0.95  # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

    ### Pretraining Config ###
    PRE_NUM_EPOCHS = 5
    PRE_TRAIN_BATCH_SIZE = 1
    PRE_BATCHES_PER_TRAIN_EPOCH = 10
    PRE_LR = 1e-4

    PRE_BATCHES_PER_VAL_EPOCH = 5
    PRE_VAL_BATCH_SIZE=1

    ### Training Config ###
    NUM_EPOCHS = 5
    TRAIN_BATCH_SIZE = 1
    BATCHES_PER_EPOCHS = 10
    LR = 1e-4
    DROPOUT = 0.2                # Should be turned off during validation/inference?

    ### Validation Config ###
    VAL_BATCH_SIZE=1

    ### Inference Config ###
    INFERENCE_BATCH_SIZE = 2