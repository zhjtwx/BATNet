import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import glob
import torch
import glob
import os
import sys
import cv2


def load_image(img_dir, idx):
    path = os.path.join(img_dir, f"{idx}.png")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(img).float() / 255.0  # 归一化到[0,1]


def get_image_mask(file, mask_file):
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
    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_file = self.img_paths[idx]
        mask_file = img_file.replace('/data_png/', '/data_mask/')
        input_img, input_mask = get_image_mask(img_file, mask_file)
        return input_img, input_mask
