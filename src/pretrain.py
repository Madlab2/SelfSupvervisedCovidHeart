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
from monai.data import DataLoader, ArrayDataset
from monai.networks.nets import UNet
from monai.networks.utils import one_hot

from monai.transforms import RandSpatialCropSamplesd

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
noisy_image = add_noise(image)
print("image noise.")

# cut through the image for validation and train split
cut_idx_z = int(np.floor(image.shape[-1]*0.7)) # 70% split 716

train_image = noisy_image[..., :cut_idx_z]
train_label = image[..., :cut_idx_z]

val_image = noisy_image[..., cut_idx_z:]
val_label = image[..., cut_idx_z:]

pretrain_dataset = ArrayDataset(data=[{'image': train_image, 'label': train_label }])
preval_dataset = ArrayDataset(val_image, val_label)

checkpoint = torch.load(convert_path('./models/worst_model_checkpoint.pth'), map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])

cropper = RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=(96, 96, 96), random_size=False, num_samples=NUM_PRE_CROPS)

train_loader = DataLoader(
    pretrain_dataset,
    batch_size=PRE_TRAIN_BATCH_SIZE,
    shuffle=False,  # Don't shuffle since we use random crops
    num_workers=0,
    pin_memory=True,
    collate_fn=cropper,
)

val_loader = DataLoader(
    preval_dataset,
    batch_size=PRE_VAL_BATCH_SIZE,
    shuffle=False,
    ### YOUR CODE HERE ###
    num_workers=0,
    pin_memory=True,
    collate_fn=cropper,
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




