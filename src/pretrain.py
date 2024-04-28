from os.path import join
from time import perf_counter as time
#from typing import Tuple, List, Dict
from tqdm import tqdm
import wandb

import numpy as np
import torch
import matplotlib.pyplot as plt
#from skimage.measure import label as skimage_label, regionprops

import monai
from monai.data import DataLoader#, Dataset
from dataset import RepeatedCacheDataset
#from monai.networks.nets import UNet
#from monai.networks.utils import one_hot

from monai.transforms import Compose, RandSpatialCropSamplesd, RandSpatialCropd, EnsureChannelFirstd

from config import *
from model import *
from utils import *



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


# NUM_OUTPUT_CHANNELS has to be 1 for the pretraining. This needs to be changed for the proper training later.
# --> before saving the pretrained model, scrap the last conv-layer (and the first skip-connect layer) 
# and replace these layers with the right output_dim and reinitialize them
# Comment: For now, we employ the channel hack

checkpoint = torch.load(convert_path(SCRATCH_MODEL), map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)

transforms = Compose([
    EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
    RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=[96, 96, 96], random_size=False, num_samples=1)
])

cropper_samples = RandSpatialCropSamplesd(keys=['image', 'label'], roi_size=[96, 96, 96], random_size=False, num_samples=1)
cropper = RandSpatialCropd(keys=['image', 'label'], roi_size=[96, 96, 96])
pretrain_dataset = RepeatedCacheDataset(data=[{'image': train_image, 'label': train_label}], 
                                        num_repeats= PRE_BATCHES_PER_TRAIN_EPOCH * PRE_TRAIN_BATCH_SIZE,
                                        transform=transforms,
                                        )

preval_dataset = RepeatedCacheDataset(data=[{'image': val_image, 'label': val_label}], 
                                        num_repeats= PRE_BATCHES_PER_VAL_EPOCH * PRE_VAL_BATCH_SIZE,
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


print("Generating figure...")
batch = next(iter(train_loader))  # Get first batch

fig, ax = plt.subplots(2, 4, figsize=(9, 4))
for i in range(4):
    ax[0, i].imshow(batch['image'][i, 0, :, :, batch['image'].shape[3] // 2], cmap='gray')
    ax[1, i].imshow(batch['label'][i, 0, :, :, batch['label'].shape[3] // 2], cmap='gray')

    if i == 0:
        ax[0, i].set_ylabel('image')
        ax[1, i].set_ylabel('label')

print("Saving figure...")
plt.savefig('./outputs/figures/train_noise_denoise.png', dpi=500)

wandb.init(
    project="jacana_sounds", # TODO add titel with pretrain
    config={'batch_size': PRE_TRAIN_BATCH_SIZE, 'num_epochs': PRE_NUM_EPOCHS, 'learning_rate':PRE_LR}
)

print('Starting pretraining')
all_train_losses = []
all_val_losses = []
best_val_loss = float('inf')
total_train_time = 0.0

for epoch in range(PRE_NUM_EPOCHS):
    mean_train_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.train()
    print(f"Epoch {epoch + 1}/{PRE_NUM_EPOCHS}")
    wandb.log({'epoch-and-time (pretrain)': epoch + 1})

    for batch in tqdm(train_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)# [1, 1, 96, 96, 96]
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True) # [1, 1, 96, 96, 96]
        
        # channel hack: make label.shape be [1, 2, 96, 96, 96] by duplicating/copying. We train both out-channels to the same target
        label = label.repeat(1, 2, 1, 1, 1)
        
        with torch.cuda.amp.autocast():
            pred = model(image_b)  # [1, 2, 96, 96, 96]
            loss = pre_loss_fn(input=pred.softmax(dim=1), target=label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=None)

        mean_train_loss += loss.detach() * len(image_b)
        num_samples += len(image_b)
        step += 1

    train_time = time() - t0
    total_train_time += train_time
    wandb.log({"epoch (pretrain)": epoch+1, 'num_train_samples (pretrain)': num_samples})

    mean_train_loss = mean_train_loss / num_samples
    all_train_losses.append(mean_train_loss.item())

    mean_val_loss = 0
    num_samples = 0
    step = 0
    t0 = time()
    model.eval()

    for batch in tqdm(val_loader):
        image_b = batch['image'].as_tensor().to(DEVICE, non_blocking=True)# [1, 1, 96, 96, 96]
        label = batch['label'].as_tensor().to(DEVICE, non_blocking=True) # [1, 1, 96, 96, 96]
        # channel hack: make label.shape be [1, 2, 96, 96, 96] by duplicating/copying. We train both out-channels to the same target
        label = label.repeat(1, 2, 1, 1, 1)

        with torch.no_grad():

            with torch.cuda.amp.autocast():
                pred = model(image_b)  # [1, 2, 96, 96, 96]
                loss = pre_loss_fn(input=pred.softmax(dim=1), target=label)

        mean_val_loss += loss * len(image_b)
        num_samples += len(image_b)
        step += 1

    val_time = time() - t0
    wandb.log({"epoch (pretrain)": epoch+1, 'num_val_samples (pretrain)': num_samples})
    mean_val_loss = mean_val_loss / num_samples
   
    wandb.log({"epoch (pretrain)": epoch+1, "train_loss (pretrain)": mean_train_loss.item(), "val_loss (pretrain)": mean_val_loss.item()})
    wandb.log({"epoch (pretrain)": epoch+1, 'train-time (pretrain)': train_time, 'total-train-time (pretrain)': total_train_time, 'val-time (pretrain)': val_time})

    # wandb log example image patch 
    pred = pred.to('cpu').numpy()
    image_plot = batch['image'].to('cpu').numpy()
    label_plot = batch['label'].to('cpu').numpy()
    fig, ax = plt.subplots(3, 4, figsize=(9, 6))
    for i in range(4):
        ax[0, i].imshow(image_plot[i, 0, :, :, image_plot.shape[3] // 2], cmap='gray')
        ax[1, i].imshow(pred[i, 0, :, :, pred.shape[3] // 2], cmap='gray')
        ax[2, i].imshow(label_plot[i, 0, :, :, label_plot.shape[3] // 2], cmap='gray')
        if i == 0:
            ax[0, i].set_ylabel('image')
            ax[1, i].set_ylabel('prediction')
            ax[2, i].set_ylabel('label')
    wandb.log({"epoch (pretrain)": epoch+1,'Noise-Denoise-Pred': fig})
    
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
        }, convert_path(f'models/pretrain/pretrain_model_checkpoint_e{epoch}_loss{mean_val_loss}.pth')) ### TODO adjust path to operating system

    print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')




