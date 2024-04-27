import os
import numpy as np
import torch
from typing import Tuple, List, Dict
from skimage.measure import label as skimage_label, regionprops

#from config import PRE_TRAIN_NOISVAR
PRE_TRAIN_NOISVAR = 0.01

def add_noise(image):
    v = (PRE_TRAIN_NOISVAR**0.5)  # variance of 0.1
    return image + v*torch.randn(image.shape)

def extract_label_patches(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> List[Dict[str, np.ndarray]]:
    """
    Extract patches from image where label is non-zero.

    For each connected component in label, extract the bounding box.
    Split the bounding box into overlapping patches of size patch_size.
    Extract the patches from image and label.
    Return the patches as a list of { 'image': ..., 'label': ... } dicts.

    Args:
        image (np.ndarray): Image to extract patches from.
        label (np.ndarray): Label to extract patches from.
        patch_size (tuple): Size of the patches to extract.

    Returns:
        list: List of patches as { 'image': ..., 'label': ..., 'mask': ... } dicts.
    """
    props = regionprops(skimage_label(label > 0))  # Extract connected components of labeled voxels
    patches = []
    for pp in props:
        # Extract bounding box for connected component
        cc_min = pp.bbox[:3]
        cc_max = pp.bbox[3:]

        # Extract patches covering the bounding box
        for z in range(cc_min[0] - patch_size[0] // 2, cc_max[0] + patch_size[0] // 2, patch_size[0] // 2):
            for y in range(cc_min[1] - patch_size[1] // 2, cc_max[1] + patch_size[1] // 2, patch_size[1] // 2):
                for x in range(cc_min[2] - patch_size[2] // 2, cc_max[2] + patch_size[2] // 2, patch_size[2] // 2):
                    # Ensure patch is within image bounds
                    z_begin = max(z, 0)
                    y_begin = max(y, 0)
                    x_begin = max(x, 0)
                    z_end = min(z + patch_size[0], image.shape[0])
                    y_end = min(y + patch_size[1], image.shape[1])
                    x_end = min(x + patch_size[2], image.shape[2])

                    patch_label = label[z_begin:z_end, y_begin:y_end, x_begin:x_end]
                    if not patch_label.any():
                        # Skip empty patches
                        continue
                    patch_image = image[z_begin:z_end, y_begin:y_end, x_begin:x_end]

                    if patch_image.shape != patch_size:
                        # Pad patch if it is smaller than patch_size
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_image.shape[d])
                        patch_image = np.pad(patch_image, pad_size, 'constant', constant_values=0)
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_label.shape[d])
                        patch_label = np.pad(patch_label, pad_size, 'constant', constant_values=0)

                    patches.append({ 'image': patch_image, 'label': patch_label, 'mask': patch_label > 0 })

    return patches


def convert_path(path):
    '''
    Converts a given Linux- filepath-string to Windows-style string if on windows
    '''
    if os.name == 'nt':
        # Windows Path
        path = path.replace('/', '\\')
    return path
