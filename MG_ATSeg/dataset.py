import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import glob
import torch
import glob
import os
import sys
import cv2


"""
In our study, the conversion from DICOM to PNG format was performed deliberately for adipose tissue segmentation. 
Unlike general-purpose CT segmentation, 
fat tissue exhibits a relatively narrow and well-defined Hounsfield Unit (HU) range. 
To ensure both computational efficiency and sufficient contrast representation, 
we restricted the HU values to the interval [-300, 200], 
which covers the entire adipose tissue distribution while excluding irrelevant soft tissue and bone densities.
All voxel intensities greater than 200 HU were clipped to 200, and those lower than -300 HU were clipped to -300. 
After clipping, the values were linearly normalized to the range [0, 1], 
then scaled by 255 and saved as 8-bit PNG images. 
"""


def load_image(img_dir, idx):
    """
    Load a grayscale image from a specified directory and index.

    Args:
        img_dir (str): Directory path containing the image files.
        idx (int): Image index (e.g., 0 -> "0.png").

    Returns:
        torch.Tensor: A normalized grayscale image tensor (range [0, 1]).
    """
    path = os.path.join(img_dir, f"{idx}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(img).float() / 255.0  # Normalize pixel values to [0,1]


def get_image_mask(file, mask_file):
    """
    Load a central image and its segmentation mask, then create
    a small sequence of neighboring slices for context.

    Args:
        file (str): Path to the center image file (e.g., ".../123.png").
        mask_file (str): Corresponding segmentation mask path.

    Returns:
        tuple:
            - sequence (torch.Tensor): Stack of 5 grayscale images
              [t-2, t-1, t, t+1, t+2], each normalized to [0,1].
              Shape: (5, H, W)
            - mask (torch.Tensor): Two-channel binary mask (background, foreground).
              Shape: (2, H, W)
    """
    inp_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    fg = torch.from_numpy(inp_mask).float() / 255.0  # [0, 1]
    bg = 1.0 - fg
    mask = torch.stack([bg, fg], dim=0)
    indices = []
    file_root = file.replace(file.split('/')[-1], '')
    img_count = len(glob.glob(file_root + '/*png')) - 1
    center_idx = int(file.split('/')[-1].split('.png')[0])
    for offset in [-2, -1, 0, 1, 2]:
        idx = center_idx + offset
        if idx < 0:
            idx = 0
        elif idx >= img_count:
            idx = img_count - 1
        indices.append(idx)
    sequence = torch.stack([load_image(file_root, i) for i in indices])
    return sequence, mask


class MyDataset(Dataset):
    """
    Custom dataset class for loading 2D medical image slices
    and their corresponding segmentation masks.

    The dataset loads each image with its neighboring context slices,
    returning both image sequence and binary mask tensors.
    """
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_file = self.img_paths[idx]
        mask_file = img_file.replace('/data_png/', '/data_mask/')
        input_img, input_mask = get_image_mask(img_file, mask_file)
        return input_img, input_mask
