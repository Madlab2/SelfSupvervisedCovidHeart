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
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.networks.utils import one_hot

from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    FgBgToIndicesd,
    LabelToMaskd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandAxisFlipd,
)

from config import *
from model import *
from dataset import RepeatedCacheDataset
from utils import *

wandb.init(
    project="jacana_sounds", # TODO add titel with pretrain
    config={'batch_size': PRE_TRAIN_BATCH_SIZE, 'num_epochs': PRE_NUM_EPOCHS, 'learning_rate':PRE_LR}
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()

print("Noising image...")
blurred_image = add_noise(image, sigma=1.0)
print("image noise.")

# cut through the image for validation and train split
cut_idx_z = np.floor(image.shape(-1)*0.7) # 70% split


train_image = blurred_image[:, :, :cut_idx_z]
train_label = image[:, :, :cut_idx_z]

val_image = blurred_image[:, :, :cut_idx_z]
val_label = image[:, :, cut_idx_z:]



pretrain_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    CopyItemsd(keys=['label'], times=1, names=['mask']),                                                  # Copy label to new image mask
    LabelToMaskd(keys=['mask'], select_labels=[1, 2], merge_channels=True),                               # Convert mask to binary mask showing where labels are
    FgBgToIndicesd(keys=['mask'], fg_postfix='_fg_indices', bg_postfix='_bg_indices'),                    # Precompute indices of labelled voxels
    RandCropByPosNegLabeld(keys=['image', 'label', 'mask'], label_key='label', spatial_size=PATCH_SIZE,   # Extract random crop
                            pos=PROB_FOREGROUND_CENTER, neg=(1.0 - PROB_FOREGROUND_CENTER),
                            num_samples=1, fg_indices_key='mask_fg_indices', bg_indices_key='mask_bg_indices'),
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 1)),                        # Randomly rotate
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(1, 2)),                        # Randomly rotate
    RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 2)),                        # Randomly rotate
    RandAxisFlipd(keys=['image', 'label', 'mask'], prob=0.1),                                             # Randomly flip
])

pretrain_dataset = RepeatedCacheDataset(
    data=[{'image': train_image, 'label': train_label}],
    num_repeats=BATCHES_PER_EPOCHS * TRAIN_BATCH_SIZE,
    transform=pretrain_transforms,
    num_workers=0,
    cache_rate=1.0,
    copy_cache=False
)

preval_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label', 'mask'], channel_dim='no_channel'),
])

preval_dataset = CacheDataset(
    data=extract_label_patches(val_image, val_label, PATCH_SIZE)[0:25],
    transform=preval_transforms,
    num_workers=0,
    cache_rate=1.0
)

checkpoint = torch.load(convert_path('./models/worst_model_checkpoint.pth'), map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])


train_loader = DataLoader(
    pretrain_dataset,
    batch_size=PRE_TRAIN_BATCH_SIZE,
    shuffle=False,  # Don't shuffle since we use random crops
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

val_loader = DataLoader(
    preval_dataset,
    batch_size=PRE_VAL_BATCH_SIZE,
    shuffle=False,
    ### YOUR CODE HERE ###
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

model.todevice(DEVICE)

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
        label = one_hot(label, num_classes=3)
        label = label[:, 1:]

        with torch.cuda.amp.autocast():     #### Probably will crash for CPU? ####
            pred = model(image_b)
            loss = pre_loss_fn(input=pred.softmax(dim=1), target=label)

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
            label = one_hot(label, num_classes=3)
            label = label[:, 1:]

            with torch.cuda.amp.autocast():     #### Probably will crash for CPU? ####
                pred = model(image_b)
                loss = pre_loss_fn(input=pred.softmax(dim=1), target=label)

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




