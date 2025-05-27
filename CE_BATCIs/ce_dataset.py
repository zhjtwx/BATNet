import glob
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import sys

sys.path.append('.')
from pro_at_patch import SymmetricalATProcessor


class BATDataset(Dataset):
    def __init__(self, root_paths, patch_size=32, seg_model_file=None, dtypes={'ct': 'float32', 'mask': 'float32'}):
        self.root_paths = [p for p in root_paths if os.path.isdir(p)]
        self.patch_size = patch_size
        self.dtypes = dtypes
        self.seg_model = None
        if seg_model_file is not None:
            self.seg_model = SymmetricalATProcessor(model_file=seg_model_file)

    def __len__(self):
        return len(self.root_paths)

    def __getitem__(self, idx):
        case_path = self.root_paths[idx]

        # 定义各文件路径
        ct_path = os.path.join(case_path, 'image.nii.gz')
        bat_path = os.path.join(case_path, 'brown_fat_mask.nii.gz')
        at_path = os.path.join(case_path, 'fat_mask.nii.gz')

        # 定义patch缓存路径
        ct_at_left_patch_path = os.path.join(case_path, 'ct_at_left_patch.nii.gz')
        ct_at_right_patch_path = os.path.join(case_path, 'ct_at_right_patch.nii.gz')
        ct_at_left_label_path = os.path.join(case_path, 'ct_at_left_label.nii.gz')
        ct_at_right_label_path = os.path.join(case_path, 'ct_at_right_label.nii.gz')

        ct_at_left_patch, ct_at_right_patch, left_label, right_label = self.extract_patches(
            ct_path, bat_path, at_path,
            ct_at_left_patch_path, ct_at_right_patch_path,
            ct_at_left_label_path, ct_at_right_label_path
        )

        label = 1 if ((np.any(left_label > 0)) or (np.any(right_label > 0))) else 0

        return {
            'left_patches': torch.tensor(ct_at_left_patch, dtype=torch.float32),  # (16, 2, 32, 32, 32)
            'right_patches': torch.tensor(ct_at_right_patch, dtype=torch.float32),
            'left_patches_label': torch.tensor(left_label, dtype=torch.float32),  # (16, 1, 32, 32, 32)
            'right_patches_label': torch.tensor(right_label, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)  # 病例级标签
        }

    def extract_patches(self, ct_path, bat_path, at_path, left_patch_path, right_patch_path, left_label_path, right_label_path):
        if os.path.exists(left_patch_path) and  os.path.exists(right_patch_path) \
                and os.path.exists(left_label_path) and os.path.exists(right_label_path):
            left_patch = nib.load(left_patch_path).get_fdata().astype(self.dtypes['ct'])
            right_patch = nib.load(right_patch_path).get_fdata().astype(self.dtypes['ct'])
            left_label = nib.load(left_label_path).get_fdata().astype(self.dtypes['mask'])
            right_label = nib.load(right_label_path).get_fdata().astype(self.dtypes['mask'])
        else:
            if self.seg_model is None:
                raise ValueError("Segmentation model is required when cached patches not available")
            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT file not found: {ct_path}")
            at_path = bat_path if os.path.exists(at_path) else None
            bat_path = bat_path if os.path.exists(bat_path) else None

            left_patch, right_patch, left_label, right_label = self.seg_model.process(ct_path, at_path, bat_path)

            # 保存生成的patch供下次使用
            nib.save(nib.Nifti1Image(left_patch, np.eye(4)), left_patch_path)
            nib.save(nib.Nifti1Image(right_patch, np.eye(4)), right_patch_path)
            nib.save(nib.Nifti1Image(left_label, np.eye(4)), left_label_path)
            nib.save(nib.Nifti1Image(right_label, np.eye(4)), right_label_path)
        return left_patch, right_patch, left_label, right_label

