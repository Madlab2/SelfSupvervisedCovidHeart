from os.path import join
from time import perf_counter as time
from typing import Tuple, List, Dict
from tqdm import tqdm
import wandb

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.measure import label as skimage_label, regionprops

import monai
from monai.data import DataLoader, Dataset
from dataset import RepeatedCacheDataset
from monai.networks.nets import UNet
#from monai.networks.utils import one_hot

from monai.transforms import Compose, RandSpatialCropSamplesd, RandSpatialCropd, EnsureChannelFirstd

from config import *

from model import *
from utils import *

wandb.init(
    project="jacana_sounds", # TODO add titel with pretrain
    config={'batch_size': PRE_TRAIN_BATCH_SIZE, 'num_epochs': PRE_NUM_EPOCHS, 'learning_rate':PRE_LR}
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()

print("Noising image...")
noisy_image = add_noise(image)
print("image noise.")

# cut through the image for validation and train split
cut_idx_z = int(np.floor(image.shape[-1]*0.7)) # 70% split 716

train_image = noisy_image[..., :cut_idx_z]
train_label = image[..., :cut_idx_z]

val_image = noisy_image[..., cut_idx_z:]
val_label = image[..., cut_idx_z:]


# TODO NUM_OUTPUT_CHANNELS has to be 1 for the pretraining. This needs to be changed for the proper training later.
# --> before saving the pretrained model, scrap the last conv-layer (and the first skip-connect layer) 
# and replace these layers with the right output_dim and reinitialize them
# Comment: For now, we employ the channel hack

checkpoint = torch.load(convert_path('./models/worst_model_checkpoint.pth'), map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])
#print("model: ", model)

transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=[96, 96, 96], random_size=False, num_samples=1)

])

cropper_samples = RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=[96, 96, 96], random_size=False, num_samples=1)
cropper = RandSpatialCropd(keys=['image', 'label'], roi_size=[96, 96, 96])
pretrain_dataset = RepeatedCacheDataset(data=[{'image': train_image, 'label': train_label}], 
                                        num_repeats= BATCHES_PER_EPOCHS * PRE_TRAIN_BATCH_SIZE,
                                        transform=transforms,
                                        )

preval_dataset = RepeatedCacheDataset(data=[{'image': val_image, 'label': val_label}], 
                                        num_repeats= BATCHES_PER_EPOCHS * PRE_VAL_BATCH_SIZE,
                                        transform=transforms,
                                        )

train_loader = DataLoader(
    pretrain_dataset,
    batch_size=PRE_TRAIN_BATCH_SIZE,
    shuffle=False,  # Don't shuffle since we use random crops
    num_workers=0,
)

val_loader = DataLoader(
    preval_dataset,
    batch_size=PRE_VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model.to(DEVICE)

print('Starting pretraining')
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')
for epoch in range(PRE_NUM_EPOCHS):
    mean_train_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.train()
    for batch in tqdm(train_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True)
        print('Label Shape: ', label.shape)
        with torch.cuda.amp.autocast():
            pred = model(image_b)
            # channel hack: network has 2 output dims, we train both to the same 1-channel target
            loss = 1/2 * pre_loss_fn(input=pred[:, 0:1, ...].softmax(dim=1), target=label) + 1/2 * pre_loss_fn(input=pred[:, 1:2, ...].softmax(dim=1), target=label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=None)

        mean_train_loss += loss.detach() * len(image_b)
        num_samples += len(image_b)
        step += 1

    train_time = time() - t0
    mean_train_loss = mean_train_loss / num_samples
    wandb.log({'mean_train_loss_pretrain': mean_train_loss})

    all_train_losses.append(mean_train_loss.item())

    mean_val_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.eval()

    for batch in tqdm(val_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True)
        
        with torch.no_grad():

            with torch.cuda.amp.autocast():     #### Probably will crash for CPU? ####
                pred = model(image_b)
                loss = 1/2 * pre_loss_fn(input=pred[:, 0:1, ...].softmax(dim=1), target=label) + 1/2 * pre_loss_fn(input=pred[:, 1:2, ...].softmax(dim=1), target=label)
        mean_val_loss += loss * len(image_b)
        num_samples += len(image_b)
        step += 1

    val_time = time() - t0
    mean_val_loss = mean_val_loss / num_samples
    wandb.log({'mean_val_loss_pretrain': mean_val_loss})

    # wandb log example image patch 
    # wandb.log({'image input': [wandb.Image(image_b.squeeze(0), caption="Input Image (blurred)")]})
    # wandb.log({'prediction': [wandb.Image(pred.squeeze(0), caption="Predicted Image (unblurred)")]})
    # wandb.log({'target': [wandb.Image(label.squeeze(0), caption="Target Image (unblurred)")]})

    all_val_losses.append(mean_val_loss.item())
    if mean_val_loss.item() < best_val_loss:
        print('Saving best model checkpoint, epoch', epoch, 'val loss', mean_val_loss.item())
        best_val_loss = mean_val_loss
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'train_losses': all_train_losses,
            'val_losses': all_val_losses,
        }, convert_path(f'models/pretrain_model_checkpoint_e{epoch}_loss{mean_val_loss}.pth')) ### TODO adjust path to operating system

    print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')




