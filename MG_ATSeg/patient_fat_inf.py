import torch
import glob
import numpy as np
import sys
import os
import pandas as pd
import cv2
import tqdm
from torch.utils.data import DataLoader

sys.path.append('./MG_ATSeg')
from model.unet_mam import MAMUnet
from dataset import MyDataset
import mts
from collections import OrderedDict
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


import torch
import numpy as np


def get_3d_bbox(pre, mask):
    nonzero_coords = np.argwhere(mask > 0)
    if len(nonzero_coords) == 0:
        return None
    z_min, y_min, x_min = nonzero_coords.min(axis=0)
    z_max, y_max, x_max = nonzero_coords.max(axis=0)
    return pre[z_min:z_max+1], mask[z_min:z_max+1]


def dice_coefficient_3d(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + smooth) / (union + smooth)


def pro_case_level_nii(nii_file):
    dcm_arr, _ = mts.load_nii(nii_file)
    dcm_arr[dcm_arr <= -300] = -300
    dcm_arr[dcm_arr >= 200] = 200
    dcm_arr = (dcm_arr + 300) / 500
    dcm_arr = dcm_arr * 255
    dcm_arr = dcm_arr.astype('uint8')
    max_len = len(dcm_arr)
    dcm_arr = torch.from_numpy(dcm_arr).float() / 255.0
    input_case = torch.zeros((max_len, 5, dcm_arr.shape[-2], dcm_arr.shape[-1]))
    for idx in range(max_len):
        if idx == 0:
            input_case[idx, 0, :, :] = dcm_arr[0, :, :]
            input_case[idx, 1, :, :] = dcm_arr[0, :, :]
            input_case[idx, 2, :, :] = dcm_arr[0, :, :]
            input_case[idx, 3, :, :] = dcm_arr[1, :, :]
            input_case[idx, 4, :, :] = dcm_arr[2, :, :]
            continue
        if idx == 1:
            input_case[idx, 0, :, :] = dcm_arr[0, :, :]
            input_case[idx, 1, :, :] = dcm_arr[1, :, :]
            input_case[idx, 2, :, :] = dcm_arr[1, :, :]
            input_case[idx, 3, :, :] = dcm_arr[2, :, :]
            input_case[idx, 4, :, :] = dcm_arr[3, :, :]
            continue
        if max_len - idx == 2:
            input_case[idx, 0, :, :] = dcm_arr[max_len - 4, :, :]
            input_case[idx, 1, :, :] = dcm_arr[max_len - 3, :, :]
            input_case[idx, 2, :, :] = dcm_arr[max_len - 2, :, :]
            input_case[idx, 3, :, :] = dcm_arr[max_len - 1, :, :]
            input_case[idx, 4, :, :] = dcm_arr[max_len - 1, :, :]
            continue
        if max_len - idx == 1:
            input_case[idx, 0, :, :] = dcm_arr[max_len - 3, :, :]
            input_case[idx, 1, :, :] = dcm_arr[max_len - 2, :, :]
            input_case[idx, 2, :, :] = dcm_arr[max_len - 1, :, :]
            input_case[idx, 3, :, :] = dcm_arr[max_len - 1, :, :]
            input_case[idx, 4, :, :] = dcm_arr[max_len - 1, :, :]
            continue
        input_case[idx, :, :, :] = dcm_arr[idx-2: idx+3, :, :]
    return input_case


def inf_case_level_at(case_nii_file, seg_model, patch_size):
    input_data = pro_case_level_nii(case_nii_file)
    chunks = torch.split(input_data, patch_size)
    d, _, w, h = input_data.size()
    seg_model.eval()
    out_arr = np.zeros((d, w, h))
    with torch.no_grad():
        for i, batch in enumerate(chunks):
            torch.cuda.empty_cache()
            batch = batch.to('cuda:0')
            case_pre = seg_model(batch).cpu().numpy()
            case_pre = case_pre[:, 1, :, :]
            case_pre = np.where(case_pre > 0.5, 1, 0)
            out_arr[patch_size*i: patch_size*i + len(batch), :, :] = case_pre
    return out_arr


def inf_main(model_file, case_nii_files, device_ids=[0, 1, 2, 3], case_at_files=None):
    loaded = torch.load(model_file, map_location='cpu')
    if isinstance(loaded, torch.nn.DataParallel):
        state_dict = loaded.module.state_dict()
    elif isinstance(loaded, torch.nn.Module):
        state_dict = loaded.state_dict()
    else:
        state_dict = loaded
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model = MAMUnet(5, 2)
    model.load_state_dict(new_state_dict)
    model = model.to('cuda:0')
    seg_model = torch.nn.DataParallel(model, device_ids=device_ids)
    for i, case_nii_file in enumerate(case_nii_files):
        case_pre = inf_case_level_at(case_nii_file, seg_model)
        if case_at_files is not None:
            mask_arr, _ = mts.load_nii(case_at_files[i])
            case_pre, mask_arr = get_3d_bbox(case_pre, mask_arr)
            print(dice_coefficient_3d(case_pre, mask_arr, smooth=1e-6))


class InfATMask:
    def __init__(self, model_file, device_ids=[0, 1]):
        self.model_file = model_file
        self.device_ids = device_ids
        self.seg_model = self.load_model(model_file, device_ids)
        self.patch_size = len(device_ids) * 2

    def load_model(self, model_file, device_ids):
        loaded = torch.load(model_file, map_location='cpu')
        if isinstance(loaded, torch.nn.DataParallel):
            state_dict = loaded.module.state_dict()
        elif isinstance(loaded, torch.nn.Module):
            state_dict = loaded.state_dict()
        else:
            state_dict = loaded
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model = MAMUnet(5, 2)
        model.load_state_dict(new_state_dict)
        model = model.to('cuda:0')
        seg_model = torch.nn.DataParallel(model, device_ids=device_ids)
        return seg_model

    def inf_case_at(self, case_nii_file):
        case_pre = inf_case_level_at(case_nii_file, self.seg_model, self.patch_size)
        return case_pre



