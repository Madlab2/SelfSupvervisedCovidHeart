from os.path import join
import sys
from time import perf_counter as time
from typing import Tuple, List, Dict
from tqdm import tqdm
import argparse

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

import wandb
from config import *


from model import *
from dataset import RepeatedCacheDataset
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train_baseline', choices=['train_baseline', 'train_with_pretrain'], help='Whether to train from scratch or using pretrained model (default: %(default)s)')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", DEVICE)

if args.mode == "train_baseline":
    print("Training from scratch")
    #MODEL = SCRATCH_MODEL
    #SAVE_PATH = './models/baseline/'

elif args.mode == "train_with_pretrain":
    print("Training using pretrained model: ", PRE_MODEL_NAME)
    MODEL = PRE_MODEL_NAME
    SAVE_PATH = './models/train_with_pretrain/'
    checkpoint = torch.load(convert_path(MODEL), map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model'])

print("Data path: ", DATA_PATH)


image = np.load(join(DATA_PATH, 'train', 'data_0.npy'))
size_x, size_y, size_z = image.shape

# if we want to slice image down, do size_x//n etc.
subset_indices = (slice(0, size_x), slice(0, size_y), slice(0, size_z))
subset_data = image[subset_indices]
image = torch.from_numpy(subset_data).float()
print("Image loaded")

train_label = np.load(join(DATA_PATH, 'train', 'mask_0.npy'))
subset_data = train_label[subset_indices]
train_label = torch.from_numpy(subset_data).float()
print("Train label loaded")

val_label = np.load(join(DATA_PATH, 'val', 'mask_0.npy'))
subset_data = val_label[subset_indices]
val_label = torch.from_numpy(subset_data).float()
print("Val label loaded")

# Loading Validation and Training Data with transforms, creating Dataloaders
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
    num_workers=0,
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
    num_workers=0,
    cache_rate=1.0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=VAL_BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # Just use the main thread for now, we just need it for visualization
)

# Loading Model to device
model.to(DEVICE)

wandb.init(
    project="jacana_sounds",
    config={'batch_size': TRAIN_BATCH_SIZE, 'num_epochs': NUM_EPOCHS, 'learning_rate':LR}
)

print('Starting training')
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')
total_train_time = 0.0

for epoch in range(NUM_EPOCHS):
    mean_train_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.train()
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    wandb.log({'epoch-and-time': epoch + 1})
    for batch in tqdm(train_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True) # shape [1, 1, 96, 96, 96]
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True)
        mask = batch['mask'].as_tensor().to(DEVICE, non_blocking=True)

        label = one_hot(label, num_classes=3)
        label = label[:, 1:]

        with torch.cuda.amp.autocast():
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
    total_train_time += train_time
    wandb.log({"epoch": epoch+1, 'num_train_samples': num_samples})
    
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

            with torch.cuda.amp.autocast():
                pred = model(image_b)
                loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

        mean_val_loss += loss * len(image_b)

        num_samples += len(image_b)
        step += 1

    val_time = time() - t0
    wandb.log({"epoch": epoch+1, 'num_val_samples': num_samples})
    
    mean_val_loss = mean_val_loss / (num_samples + 1e-9)
    all_val_losses.append(mean_val_loss.item())

    wandb.log({"epoch": epoch+1, "train_loss": mean_train_loss.item(), "val_loss": mean_val_loss.item()})
    wandb.log({"epoch": epoch+1, 'train-time': train_time, 'total-train-time':total_train_time, 'val-time': val_time})

    # wandb log example image patch
    pred = pred.to('cpu').numpy()
    image_plot = batch['image'].to('cpu').numpy()
    label_plot = batch['label'].to('cpu').numpy()
    fig, ax = plt.subplots(3, 4, figsize=(9, 6))
    for i in range(4):
        ax[0, i].imshow(image_plot[i, 0, :, :, image_plot.shape[3] // 2], cmap='gray')
        ax[1, i].imshow(image_plot[i, 0, :, :, image_plot.shape[3] // 2], cmap='gray')
        ax[1, i].imshow(pred[i, 0, :, :, pred.shape[3] // 2], alpha=0.4)
        ax[2, i].imshow(label_plot[i, 0, :, :, label_plot.shape[3] // 2])
        if i == 0:
            ax[0, i].set_ylabel('image')
            ax[1, i].set_ylabel('image + prediction')
            ax[2, i].set_ylabel('label')
    wandb.log({"epoch": epoch+1,'Segmentation-Pred': fig})
    
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
        }, convert_path(SAVE_PATH + f'model_checkpoint_e{epoch}_loss{mean_val_loss}.pth'))

    print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')




