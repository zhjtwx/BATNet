#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BAT Classification Inference Script

Usage:
    python infer.py \
        --input /path/to/case_or_csv \
        --cls_weight ./model_weights/ce_batcis_model.pth \
        --seg_weight ./model_weights/mg_atseg_model.pth \
        --output result.csv

Examples:
    # Single case
    python infer.py --input ./data/case001

    # Multiple cases from CSV
    python infer.py --input ./data/info.csv

    # Multiple case
    python infer.py --input ./data/case001, ./data/case002 ...
"""

import os
import sys
import argparse
import multiprocessing as mp
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append('.')
sys.path.append('./CE_BATCIs')

from model.resnet_cim import CEBATCls
from ce_dataset import BATDataset


class BATInference:
    def __init__(
        self,
        cls_model_path,
        seg_model_path=None,
        device='cuda',
        patch_size=32
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )

        self.patch_size = patch_size
        self.seg_model_path = seg_model_path

        self.model = self._load_model(cls_model_path)
        self.model.eval()

    def _load_model(self, model_path):
        model = CEBATCls()

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(self.device)
        return model

    @torch.no_grad()
    def predict(self, case_dirs, batch_size=32, num_workers=6):
        dataset = BATDataset(
            root_paths=case_dirs,
            patch_size=self.patch_size,
            seg_model_file=self.seg_model_path,
            save_nii=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        predictions = []

        for batch in dataloader:
            left = batch['left_patches'].to(
                self.device, non_blocking=True
            )
            right = batch['right_patches'].to(
                self.device, non_blocking=True
            )

            logits, _ = self.model(left, right)
            probs = torch.softmax(logits, dim=1)[:, 1]

            predictions.extend(
                probs.cpu().numpy().tolist()
            )

        return predictions


def parse_input_paths(inputs):
    """
    Supported input formats:

    1. CSV file
       - Must contain a 'case_id' column.

    2. Single root directory
       - Recursively scans all valid case folders.

    3. Multiple case directories
       - Filters and keeps only valid cases.

    A valid case directory must satisfy one of the following:

    A. Contains both:
       - image.nii.gz
       - lobe.nii.gz

    B. Contains both:
       - ct_at_left_patch.nii.gz
       - ct_at_right_patch.nii.gz
    """
    def is_valid_case(case_dir):
        """
        Check whether a directory is a valid BAT case.
        """
        if not os.path.isdir(case_dir):
            return False

        # Condition 1: Original CT + lung mask
        cond1 = (
            os.path.exists(os.path.join(case_dir, 'image.nii.gz')) and
            os.path.exists(os.path.join(case_dir, 'lobe.nii.gz'))
        )

        # Condition 2: Pre-extracted BAT patches
        cond2 = (
            os.path.exists(
                os.path.join(case_dir, 'ct_at_left_patch.nii.gz')
            ) and
            os.path.exists(
                os.path.join(case_dir, 'ct_at_right_patch.nii.gz')
            )
        )

        return cond1 or cond2

    def scan_case_dirs(root_dir):
        """
        Recursively scan all directories under the given root directory.

        Every directory encountered during traversal will be checked.
        If it satisfies the BAT inference requirements, it will be
        added to the valid case list.
        """
        valid_cases = []

        if not os.path.isdir(root_dir):
            raise ValueError(f'Invalid directory: {root_dir}')

        for current_root, _, _ in os.walk(root_dir):
            if is_valid_case(current_root):
                valid_cases.append(os.path.abspath(current_root))

        return sorted(list(set(valid_cases)))

    # Single input
    # print(inputs)
    if len(inputs) == 1:
        input_path = os.path.abspath(inputs[0])

        # CSV input
        if os.path.isfile(input_path):
            if not input_path.lower().endswith('.csv'):
                raise ValueError(
                    f'Unsupported file format: {input_path}'
                )

            df = pd.read_csv(input_path)

            if 'case_id' not in df.columns:
                raise ValueError(
                    f'CSV must contain a "case_id" column: {input_path}'
                )

            case_dirs = df['case_id'].dropna().tolist()

            valid_cases = [
                case_dir
                for case_dir in case_dirs
                if is_valid_case(case_dir)
            ]

            print(
                f'Found {len(case_dirs)} cases in CSV, '
                f'{len(valid_cases)} valid.'
            )

            return valid_cases

        # Directory input
        if os.path.isdir(input_path):
            valid_cases = scan_case_dirs(input_path)

            if len(valid_cases) == 0:
                raise ValueError(
                    f'No valid cases found under:\n{input_path}'
                )

            print(
                f'Found {len(valid_cases)} valid cases.'
            )

            return valid_cases

    # Multiple directory inputs
    valid_cases = []

    for path in inputs:
        path = os.path.abspath(path)

        if is_valid_case(path):
            valid_cases.append(path)
        else:
            print(f'[Skipped] Invalid case directory: {path}')

    if len(valid_cases) == 0:
        raise ValueError(
            'No valid case directories found.'
        )

    print(
        f'Found {len(valid_cases)} valid cases.'
    )

    return sorted(valid_cases)


def main():
    parser = argparse.ArgumentParser(
        description='BAT Classification Inference'
    )

    parser.add_argument(
        '--input',
        nargs='+',
        required=True,
        help='Case directory, multiple directories, or CSV file'
    )

    parser.add_argument(
        '--cls_weight',
        default='./model_weights/ce_batcis_model.pth',
        help='Classification model path'
    )

    parser.add_argument(
        '--seg_weight',
        default='./model_weights/mg_atseg_model.pth',
        help='Segmentation model path'
    )

    parser.add_argument(
        '--device',
        default='cuda',
        help='cuda or cpu'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=6
    )

    parser.add_argument(
        '--output',
        default=None,
        help='Optional CSV output path'
    )

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)

    case_dirs = parse_input_paths(args.input)

    predictor = BATInference(
        cls_model_path=args.cls_weight,
        seg_model_path=args.seg_weight,
        device=args.device
    )

    predictions = predictor.predict(
        case_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    result_df = pd.DataFrame({
        'case_id': case_dirs,
        'prediction': predictions
    })

    print("\nInference Results")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print("=" * 80)

    if args.output:
        result_df.to_csv(args.output, index=False)
        print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
