from monai.inferers import sliding_window_inference
from dataset import RepeatedCacheDataset
from utils import extract_label_patches
from model import model, PATCH_SIZE, loss_fn
import torch
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from skimage.io import imsave
import nibabel as nib
from config import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

image = torch.from_numpy(np.load(join(DATA_PATH, 'train', 'data_0.npy'))).float()

checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model'])

model.eval()
with torch.no_grad():
    # Evaluate the model on the image using MONAI sliding window inference
    pred = sliding_window_inference(
        image[None, None],
        PATCH_SIZE,
        INFERENCE_BATCH_SIZE,
        lambda x: model(x.cuda()).softmax(dim=1).cpu(),  # send patch to GPU, run model, call softmax, send result back to CPU
        overlap=WINDOW_OVERLAP,
        mode='gaussian',
        progress=True,
    )
pred = pred.numpy()

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(image[:, :, image.shape[-1] // 2], cmap='gray')
ax[0].imshow(pred[0, 0, :, :, pred.shape[-1] // 2], alpha=0.4)
ax[1].imshow(image[:, image.shape[-2] // 2, :], cmap='gray')
ax[1].imshow(pred[0, 0, :, pred.shape[-2] // 2, :], alpha=0.4)
ax[2].imshow(image[image.shape[-3] // 2, :, :], cmap='gray')
ax[2].imshow(pred[0, 0, pred.shape[-3] // 2, :, :], alpha=0.4)
plt.show()

pred = np.uint8(pred[0, 0] * 255)

# Pick one
#imsave('pred.tiff', pred)  # For paraview
nib.save(nib.Nifti1Image(pred, np.eye(4)), 'outputs/pred.nii.gz') # For ITK-SNAP
#np.save('pred.npy', pred)  # For TomViz