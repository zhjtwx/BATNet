import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from pro_at_patch import SymmetricalATProcessor
sys.path.append('./CE_BATCIs')
from model.resnet_cim import CEBATCls
from ce_dataset import BATDataset
from torch.utils.data import DataLoader
import pandas as pd
from glob import glob
import os
import multiprocessing as mp


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
class BATInference:
    def __init__(self, cls_model_path, device='cuda', seg_model_path=None,
                 dtypes={'ct': 'float32', 'mask': 'float32'}):
        """
        A class for running inference using a trained BAT classification model
        and (optionally) a segmentation model for automatic patch extraction.

        Attributes:
            model (torch.nn.Module): The trained classification model.
            device (torch.device): The computation device ('cuda' or 'cpu').
            seg_model_path (str): Path to the segmentation model.
            patch_size (int): The size of the input patches (default: 32).
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_cls_model(cls_model_path, self.device)

        self.model.eval()
        self.patch_size = 32
        self.dtypes = dtypes
        self.seg_model_path = seg_model_path

    def load_cls_model(self, cls_model_path, device):
        """
        Load the classification model and its parameters.

        Args:
            cls_model_path (str): Path to the classification model checkpoint (.pth).
            device (torch.device): Target device for model loading.

        Returns:
            model (torch.nn.Module): The initialized classification model.
        """
        model = CEBATCls().to(device)
        case_state_dict = torch.load(cls_model_path, map_location=device)
        new_state_dict = {}
        for k, v in case_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(device)
        return model

    def batch_predict(self, case_dirs):
        """
        Perform batch inference on multiple cases.

        Args:
            case_dirs (list): List of directories, each containing one case’s NIfTI files.

        Returns:
            np.ndarray: Array of case-level prediction probabilities for the positive class.
        """
        infer_dataset = BATDataset(
            root_paths=case_dirs,
            patch_size=32,
            seg_model_file=self.seg_model_path,
            save_nii=False
        )

        infer_loader = DataLoader(
            infer_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=6
        )
        self.model.eval()
        case_preds = []
        with torch.no_grad():
            for batch in infer_loader:
                left = batch['left_patches'].to(self.device)
                right = batch['right_patches'].to(self.device)
                case_pred, _ = self.model(left, right)
                case_preds.append(torch.softmax(case_pred, dim=1)[:, 1].view(-1).cpu())
        case_preds = torch.cat(case_preds).numpy()
        return case_preds


def infer_main(case_dirs):
    mp.set_start_method('spawn', force=True)
    predictor = BATInference(
        cls_model_path='./model_weights/ce_batcis_model.pth',
        device='cuda',
        seg_model_path='./model_weights/mg_atseg_model.pth'
    )
    out_pres = predictor.batch_predict(case_dirs)
    return out_pres


def nii_externa_testset_10():
    df = pd.read_csv('./data/nii_10/info.csv')
    case_dirs = df['case_id'].tolist()
    out_pres = infer_main(case_dirs)
    df['pre'] = out_pres
    df.to_csv('./data/nii_10/infer_info.csv', index=False)


def infer_internal_testset_182():
    df = pd.read_csv('./data/crop_internal_testset_182/info.csv')
    case_dirs = df['case_id'].tolist()
    out_pres = infer_main(case_dirs)
    df['pre'] = out_pres
    df.to_csv('./data/crop_internal_testset_182/infer_info.csv', index=False)

def infer_externa_testset_744():
    df = pd.read_csv('./data/crop_744/info.csv')
    case_dirs = df['case_id'].tolist()
    out_pres = infer_main(case_dirs)
    df['pre'] = out_pres
    df.to_csv('./data/crop_744/infer_info.csv', index=False)

def infer_lzy_case_14_nii():
    df = pd.read_csv('./data/nii_lzy_testset_28/info.csv')
    case_dirs = df['case_id'].tolist()
    out_pres = infer_main(case_dirs)
    df['pre'] = out_pres
    df.to_csv('./data/nii_lzy_testset_28/infer_info.csv', index=False)


def infer_nc_case_252_nii():
    df = pd.read_csv('./data/nii_nc_testset_252/info.csv')
    case_dirs = df['case_id'].tolist()
    out_pres = infer_main(case_dirs)
    df['pre'] = out_pres
    df.to_csv('./data/nii_nc_testset_252/infer_info.csv', index=False)


if __name__ == "__main__":
    nii_externa_testset_10()
    infer_externa_testset_744()





