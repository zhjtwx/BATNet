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
        Args:
            model: 预加载的AT分割模型 (需实现predict方法)
            patch_size: 输出块大小 (默认32×32×32)
            partitions: 分区方案 (z,x,y轴) 默认(2,4,4)对应32个块
        """
        if model_file is not None:
            self.seg_at_model = InfATMask(model_file, device_ids=device_ids)
        else:
            self.seg_at_model = None
        self.patch_size = patch_size
        self.partitions = partitions  # (n_z, n_x, n_y)
        self.window_level = -100  # CT窗宽窗位
        self.window_width = 400

    def preprocess_ct(self, ct_vol):
        """CT值标准化 (HU窗口化)"""
        ct_vol = np.clip(ct_vol, self.window_level - self.window_width/2,
                         self.window_level + self.window_width/2)
        return (ct_vol - self.window_level) / self.window_width


    def get_fat_bounding_box(self, at_mask, padding=0.01):
        fat_voxels = np.where(at_mask > 0.5)  # 获取所有脂肪体素坐标
        if len(fat_voxels[0]) == 0:
            raise ValueError("AT mask中未检测到脂肪体素")

        # 计算最小/最大坐标
        min_coords = np.array([np.min(axis) for axis in fat_voxels])
        max_coords = np.array([np.max(axis) for axis in fat_voxels])

        # 添加padding（确保不越界）
        size = max_coords - min_coords
        pad = (size * padding).astype(int)
        min_coords = np.maximum(0, min_coords - pad)
        max_coords = np.minimum(at_mask.shape, max_coords + pad)

        return tuple(min_coords), tuple(max_coords)

    def split_into_32_patches(self, subvol, target_size=32, grid=(2, 2, 8)):
        """将子体积均匀切分为2(z)×4(x)×4(y)=32块并重采样"""
        n_z, n_x, n_y = grid
        left_patches = {}
        right_patches = {}
        # 计算每个维度的切分位置（使用ceil确保覆盖整个体积）
        z_splits = np.linspace(0, subvol.shape[0], n_z + 1, dtype=int)
        x_splits = np.linspace(0, subvol.shape[1], n_x + 1, dtype=int)
        y_splits = np.linspace(0, subvol.shape[2], n_y + 1, dtype=int)
        # 三维循环切块
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

    def process(self, ct_nii_path, at_nii_path=None, bat_nii_path=None):
        """
        主处理流程:
        1. 使用模型生成AT mask
        2. 提取边界长方体
        3. 分区+重采样
        4. 对称分割

        返回: ((left_ct, right_ct), (left_at, right_at))
        """
        # 1. 加载CT并生成AT mask
        ct_volume = nib.load(ct_nii_path).get_fdata()
        ct_volume = self.preprocess_ct(ct_volume)
        if at_nii_path is not None:
            at_mask = nib.load(at_nii_path).get_fdata().astype(np.uint8)
            if at_mask.shape != ct_volume.shape:
                at_mask = self.seg_at_model.inf_case_at(ct_nii_path).astype(np.uint8)
                at_mask[at_mask.shape[0]//3:, :, :] = 0
        else:
            at_mask = self.seg_at_model.inf_case_at(ct_nii_path).astype(np.uint8)
            at_mask[at_mask.shape[0] // 3:, :, :] = 0

        # 2. 提取边界长方体
        (z0, x0, y0), (z1, x1, y1) = self.get_fat_bounding_box(at_mask)
        if bat_nii_path is not None:
            bt_mask = nib.load(bat_nii_path).get_fdata().astype(np.uint8)
            if bt_mask.shape != ct_volume.shape:
                bt_mask = np.zeros(ct_volume.shape).astype(np.uint8)
        else:
            bt_mask = np.zeros(ct_volume.shape).astype(np.uint8)
        ct_subvol = ct_volume[z0:z1, x0:x1, y0:y1]
        at_subvol = at_mask[z0:z1, x0:x1, y0:y1]
        bt_subvol = bt_mask[z0:z1, x0:x1, y0:y1]
        # 3. 分区处理
        ct_left_patches, ct_right_patches = self.split_into_32_patches(ct_subvol)
        at_left_patches, at_right_patches = self.split_into_32_patches(at_subvol)
        bt_left_patches, bt_right_patches = self.split_into_32_patches(bt_subvol)
        left_label = generate_labels(at_left_patches, bt_left_patches)
        right_label = generate_labels(at_right_patches, bt_right_patches)

        ct_at_left = np.stack([ct_left_patches, at_left_patches], axis=1)
        ct_at_right = np.stack([ct_right_patches, at_right_patches], axis=1)
        return ct_at_left, ct_at_right, left_label, right_label


def generate_labels(at_mask, bat_mask, threshold=0.1):
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


def save_nii(data,  filename):
    """保存NIfTI文件并确保目录存在"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=np.eye(4)), filename)

# if __name__ == "__main__":
#     save_fp = '/data/tanweixiong/dongzhong/zongsezhifang/BATNet/data'
#     seg_model_file = '/data/tanweixiong/dongzhong/zongsezhifang/BATNet/MG_ATSeg/save_seg_model/seg_best.pth'
#     processor = SymmetricalATProcessor(model_file=seg_model_file)
#     fat_file_list = glob.glob('/data/tanweixiong/dongzhong/zongsezhifang/0310_fat_brown_fat/*/*/*/fat_mask.nii.gz')
#     ii = 0
#     for fat_file in fat_file_list:
#         image_file = fat_file.replace('fat_mask.nii.gz', 'image.nii.gz')
#         bat_file = fat_file.replace('fat_mask.nii.gz', 'brown_fat_mask.nii.gz')
#         sub_dir = '/'.join(fat_file.split('/')[-4:-1])
#
#         ct_at_left, ct_at_right, left_label, right_label = processor.process(image_file, fat_file, bat_file)
#         if (left_label == right_label).all():
#             ii = ii + 1
#             print(ii)
#             image_file_fp = os.path.join(save_fp, sub_dir, 'image.nii.gz')
#             os.makedirs(os.path.dirname(image_file_fp), exist_ok=True)
#             shutil.copy2(image_file, image_file_fp)
#
#             fat_file_fp = os.path.join(save_fp, sub_dir, 'fat_mask.nii.gz')
#             os.makedirs(os.path.dirname(fat_file_fp), exist_ok=True)
#             shutil.copy2(fat_file, fat_file_fp)
#
#             bat_file_fp = os.path.join(save_fp, sub_dir, 'brown_fat_mask.nii.gz')
#             os.makedirs(os.path.dirname(bat_file_fp), exist_ok=True)
#             shutil.copy2(bat_file, bat_file_fp)
#
#             ct_at_left_fp = os.path.join(save_fp, sub_dir, 'ct_at_left_patch.nii.gz')
#             save_nii(ct_at_left, ct_at_left_fp)
#
#             ct_at_right_fp = os.path.join(save_fp, sub_dir, 'ct_at_right_patch.nii.gz')
#             save_nii(ct_at_right, ct_at_right_fp)
#
#             ct_at_left_fp = os.path.join(save_fp, sub_dir, 'ct_at_left_label.nii.gz')
#             save_nii(left_label, ct_at_left_fp)
#
#             ct_at_right_fp = os.path.join(save_fp, sub_dir, 'ct_at_right_label.nii.gz')
#             save_nii(right_label, ct_at_right_fp)

