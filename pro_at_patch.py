import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from scipy.ndimage import zoom
import glob
import os
import sys
import shutil
sys.path.append('./MG_ATSeg')
from patient_fat_inf import InfATMask


class SymmetricalATProcessor:
    def __init__(self, patch_size=32, partitions=(2, 4, 4), model_file=None, device_ids=[0]):
        """
                Symmetrical Adipose Tissue Processor for extracting and analyzing bilateral fat patches

                Args:
                    patch_size: Output patch size (default 32×32×32)
                    partitions: Partition scheme (z,x,y axes) default (2,4,4) for 32 patches
                    model_file: Path to pre-trained AT segmentation model
                    device_ids: GPU device IDs for model inference
        """
        if model_file is not None:
            self.seg_at_model = InfATMask(model_file, device_ids=device_ids)
        else:
            self.seg_at_model = None
        self.patch_size = patch_size
        self.partitions = partitions  # (n_z, n_x, n_y)
        self.window_level = -100  # CT window level
        self.window_width = 400  # CT window width

    def preprocess_ct(self, ct_vol):
        """CT value normalization (HU windowing)"""
        ct_vol = np.clip(ct_vol, self.window_level - self.window_width/2,
                         self.window_level + self.window_width/2)
        return (ct_vol - self.window_level) / self.window_width


    def get_fat_bounding_box(self, at_mask, padding=0.01):
        """Extract bounding box containing adipose tissue with padding"""
        fat_voxels = np.where(at_mask > 0.5)
        if len(fat_voxels[0]) == 0:
            return tuple([1, 126, 126]), tuple([89, 330, 340])
            raise ValueError("AT mask中未检测到脂肪体素")

        min_coords = np.array([np.min(axis) for axis in fat_voxels])
        max_coords = np.array([np.max(axis) for axis in fat_voxels])

        size = max_coords - min_coords
        pad = (size * padding).astype(int)
        min_coords = np.maximum(0, min_coords - pad)
        max_coords = np.minimum(at_mask.shape, max_coords + pad)

        return tuple(min_coords), tuple(max_coords)

    def split_into_32_patches(self, subvol, target_size=32, grid=(2, 2, 8)):
        """Split subvolume into 32 symmetrical patches (16 left, 16 right)"""
        n_z, n_x, n_y = grid
        left_patches = {}
        right_patches = {}
        z_splits = np.linspace(0, subvol.shape[0], n_z + 1, dtype=int)
        x_splits = np.linspace(0, subvol.shape[1], n_x + 1, dtype=int)
        y_splits = np.linspace(0, subvol.shape[2], n_y + 1, dtype=int)

        for idz, (z0, z1) in enumerate(zip(z_splits[:-1], z_splits[1:])):
            for idx, (x0, x1) in enumerate(zip(x_splits[:-1], x_splits[1:])):
                for idy, (y0, y1) in enumerate(zip(y_splits[:-1], y_splits[1:])):
                    key_idx = '_'.join([str(idz), str(idx), str(idy)])
                    patch = subvol[z0:z1, x0:x1, y0:y1]
                    zoom_factor = [target_size / s for s in patch.shape]
                    resized_patch = zoom(patch, zoom_factor,
                                         order=0 if patch.dtype == np.uint8 else 1)
                    if idy < 4:
                        left_patches[key_idx] = resized_patch
                    else:
                        right_patches[key_idx] = resized_patch
        match_left_patches = []
        match_right_patches = []
        for k_l, v_l in left_patches.items():
            l_z, l_x, l_y = int(k_l.split('_')[0]), int(k_l.split('_')[1]), int(k_l.split('_')[2])
            for k_r, v_r in right_patches.items():
                r_z, r_x, r_y = int(k_r.split('_')[0]), int(k_r.split('_')[1]), int(k_r.split('_')[2])
                if l_z == r_z and l_x == r_x and l_y + r_y == 7:
                    match_left_patches.append(v_l), match_right_patches.append(v_r)
        return np.array(match_left_patches), np.array(match_right_patches)

    def process(self, ct_nii_path, at_nii_path=None, bat_nii_path=None, lobe_nii_path=None):
        """Main processing pipeline for symmetrical AT analysis"""
        ct_volume = nib.load(ct_nii_path).get_fdata()
        ct_volume = self.preprocess_ct(ct_volume)
        if at_nii_path is not None and os.path.exists(at_nii_path):
            at_mask = nib.load(at_nii_path).get_fdata().astype(np.uint8)
            if at_mask.shape != ct_volume.shape:
                if self.seg_at_model is not None:
                    at_mask = self.seg_at_model.inf_case_at(ct_nii_path).astype(np.uint8)
                    at_mask[at_mask.shape[0]//3:, :, :] = 0
                else:
                    raise ValueError("Segmentation model is required when cached patches not available")
        else:
            if self.seg_at_model is not None:
                at_mask = self.seg_at_model.inf_case_at(ct_nii_path).astype(np.uint8)
                # at_mask[at_mask.shape[0] // 3:, :, :] = 0
            else:
                raise ValueError("Segmentation model is required when cached patches not available")
        if lobe_nii_path is not None:
            lobe_nii = nib.load(lobe_nii_path).get_fdata().astype(np.uint8)
            min_coords, max_coords = self.get_fat_bounding_box(lobe_nii)
            at_mask[(max_coords[0] + min_coords[0]) * 3 // 5:, :, :] = 0
            save_nii(at_mask, lobe_nii_path.replace('/lobe.nii.gz', '/seg_at_mask.nii.gz'))

        (z0, x0, y0), (z1, x1, y1) = self.get_fat_bounding_box(at_mask)

        if bat_nii_path is not None and os.path.exists(bat_nii_path):
            bt_mask = nib.load(bat_nii_path).get_fdata().astype(np.uint8)
            if bt_mask.shape != ct_volume.shape:
                bt_mask = np.zeros(ct_volume.shape).astype(np.uint8)
        else:
            bt_mask = np.zeros(ct_volume.shape).astype(np.uint8)
        ct_subvol = ct_volume[z0:z1, x0:x1, y0:y1]
        at_subvol = at_mask[z0:z1, x0:x1, y0:y1]
        bt_subvol = bt_mask[z0:z1, x0:x1, y0:y1]


        ct_left_patches, ct_right_patches = self.split_into_32_patches(ct_subvol)

        at_left_patches, at_right_patches = self.split_into_32_patches(at_subvol)
        bt_left_patches, bt_right_patches = self.split_into_32_patches(bt_subvol)
        left_label = generate_labels(at_left_patches, bt_left_patches)
        right_label = generate_labels(at_right_patches, bt_right_patches)

        ct_at_left = np.stack([ct_left_patches, at_left_patches], axis=1)
        ct_at_right = np.stack([ct_right_patches, at_right_patches], axis=1)
        return ct_at_left, ct_at_right, left_label, right_label


def generate_labels(at_mask, bat_mask, threshold=0.1):
    """Generate labels based on BAT ratio in AT patches"""
    at_mask = (at_mask > 0).astype(int)
    bat_mask = (bat_mask > 0).astype(int)
    labels = np.full(at_mask.shape[0], -1)
    for i in range(at_mask.shape[0]):
        at_voxels = np.sum(at_mask[i])
        bat_voxels = np.sum(bat_mask[i] * at_mask[i])
        if at_voxels == 0:
            ratio = 0
        else:
            ratio = bat_voxels / at_voxels
        if ratio == 0:
            labels[i] = 0
        elif ratio >= threshold:
            labels[i] = 1
    return labels


def save_nii(data, filename):
    """Save NIfTI file and ensure directory exists"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), filename)




