from os.path import join
from time import perf_counter as time
from typing import Tuple, List, Dict
from tqdm import tqdm

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
from utils import extract_label_patches


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()
train_label = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'mask_0.npy')))
val_label = torch.from_numpy(np.load(join(DATA_PATH, 'val', 'mask_0.npy')))

train_transforms = Compose([
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

train_dataset = RepeatedCacheDataset(
    data=[{ 'image': image, 'label': train_label }],
    num_repeats=BATCHES_PER_EPOCHS * TRAIN_BATCH_SIZE,
    transform=train_transforms,
    num_workers=8,
    cache_rate=1.0,
    copy_cache=False  # Important to avoid slowdowns
)

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,  # No need to shuffle since we repeat the data
    num_workers=0,  # Just use the main thread for now, we just need it for visualization
)

val_transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label', 'mask'], channel_dim='no_channel'),
])

val_dataset = CacheDataset(
    data=extract_label_patches(image, val_label, PATCH_SIZE),
    transform=val_transforms,
    num_workers=8,
    cache_rate=1.0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # Just use the main thread for now, we just need it for visualization
)

checkpoint = torch.load('/models/worst_model_checkpoint.pth')
model.load_state_dict(checkpoint['model'])


train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=False,  # Don't shuffle since we use random crops
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    ### YOUR CODE HERE ###
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)

model.todevice(DEVICE)

print('Starting training')
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    mean_train_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.train()
    for batch in tqdm(train_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True)
        mask = batch['mask'].as_tensor().to(DEVICE, non_blocking=True)
        label = one_hot(label, num_classes=3)
        label = label[:, 1:]

        with torch.cuda.amp.autocast():     #### Probably will crash for CPU? ####
            pred = model(image_b)
            loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=None)

        mean_train_loss += loss.detach() * len(image_b)
        num_samples += len(image_b)
        step += 1

    train_time = time() - t0
    mean_train_loss = mean_train_loss / num_samples
    all_train_losses.append(mean_train_loss.item())

    mean_val_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.eval()
    for batch in tqdm(val_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True)
        mask = batch['mask'].as_tensor().to(DEVICE, non_blocking=True)
        
        with torch.no_grad():
            label = one_hot(label, num_classes=3)
            label = label[:, 1:]

            with torch.cuda.amp.autocast():     #### Probably will crash for CPU? ####
                pred = model(image_b)
                loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

        mean_val_loss += loss * len(image_b)
        num_samples += len(image_b)
        step += 1

    val_time = time() - t0
    mean_val_loss = mean_val_loss / num_samples
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
        }, f'models/model_checkpoint_e{epoch}.pth') ### TODO adjust path to operating system

    print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')



