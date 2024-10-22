import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.io import imsave
from monai.inferers import sliding_window_inference

from os.path import join

from config import *
from model import *
from dataset import RepeatedCacheDataset
from utils import *


PRE_TRAIN = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Inference: loading image...")
image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float().to(DEVICE)

# specify best model here:
BEST_MODEL_WITH_PRETRAIN = './models/train_with_pretrain/real_scratch/model_checkpoint_e87_loss0.34498000144958496.pth'
BEST_MODEL_SCRATCH = './models/baseline/real_scratch/model_checkpoint_e95_loss0.3363226056098938.pth'

if PRE_TRAIN:
    BEST_MODEL = BEST_MODEL_WITH_PRETRAIN
else:
    BEST_MODEL = BEST_MODEL_SCRATCH

print(f"Inference: loading mode from {BEST_MODEL}")
checkpoint = torch.load(BEST_MODEL, map_location=torch.device(DEVICE))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE)

print("Inference: evaluating...")
model.eval()
with torch.no_grad():
    # Evaluate the model on the image using MONAI sliding window inference
    pred = sliding_window_inference(
        image[None, None],
        PATCH_SIZE,
        INFERENCE_BATCH_SIZE,
        lambda x: model(x.cuda()).softmax(dim=1),  # send patch to GPU, run model, call softmax, send result back to CPU
        overlap=WINDOW_OVERLAP,
        mode='gaussian',
        progress=True,
    )


pred = pred.cpu().numpy()
image = image.cpu().numpy()

print("Inference: creating plots...")
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
ax[0, 0].imshow(image[:, :, image.shape[-1] // 2], cmap='gray')
ax[0, 0].imshow(pred[0, 0, :, :, pred.shape[-1] // 2], alpha=0.4)

ax[1, 0].imshow(image[:, :, image.shape[-1] // 2], cmap='gray')
ax[1, 0].imshow(pred[0, 1, :, :, pred.shape[-1] // 2], alpha=0.4)

ax[0, 1].imshow(image[:, image.shape[-2] // 2, :], cmap='gray')
ax[0, 1].imshow(pred[0, 0, :, pred.shape[-2] // 2, :], alpha=0.4)

ax[1, 1].imshow(image[:, image.shape[-2] // 2, :], cmap='gray')
ax[1, 1].imshow(pred[0, 1, :, pred.shape[-2] // 2, :], alpha=0.4)

ax[0, 2].imshow(image[image.shape[-3] // 2, :, :], cmap='gray')
ax[0, 2].imshow(pred[0, 0, pred.shape[-3] // 2, :, :], alpha=0.4)

ax[1, 2].imshow(image[image.shape[-3] // 2, :, :], cmap='gray')
ax[1, 2].imshow(pred[0, 1, pred.shape[-3] // 2, :, :], alpha=0.4)

if PRE_TRAIN:
    plt.suptitle("With Pretrain - Inference on Full Image")
    plt.savefig('./outputs/figures/train_with_pretrain/pre_final.png', dpi=500)
else:
    plt.suptitle("No Pretrain - Inference on Full Image")
    plt.savefig('./outputs/figures/baseline/baseline_final.png', dpi=500)

pred = np.uint8(pred[0, 0] * 255) # [1024 1024 1024]

# Pick one
#imsave('pred.tiff', pred)  # For paraview
if PRE_TRAIN:
    nib.save(nib.Nifti1Image(pred, np.eye(4)), convert_path('./outputs/files/train_with_pretrain/pre_final.nii.gz')) # For ITK-SNAP
    np.save('./outputs/files/train_with_pretrain/pre_final.npy', pred)  # For TomViz
else:
    nib.save(nib.Nifti1Image(pred, np.eye(4)), convert_path('./outputs/files/baseline/baseline_final.nii.gz')) # For ITK-SNAP
    np.save('./outputs/files/baseline/baseline_final.npy', pred)  # For TomViz