# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('./CE_BATCIs')
from model.resnet_cim import CEBATCls
from ce_dataset import BATDataset
import mts
from torch.optim import lr_scheduler
import numpy as np
import os
import glob
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.case_criterion = nn.BCEWithLogitsLoss()
        self.patch_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.scaler = GradScaler()

        self.best_val_auc = 0

    def train_epoch(self):
        self.model.train()
        total_case_loss = 0
        total_patch_loss = 0
        for batch in self.train_loader:
            left = batch['left_patches'].to(self.device)  # (B,16,2,32,32,32)
            right = batch['right_patches'].to(self.device)
            left_labels = batch['left_patches_label'].to(self.device)  # (B,16)
            right_labels = batch['right_patches_label'].to(self.device)
            case_labels = batch['label'].to(self.device)  # (B,)

            patch_labels = torch.cat([
                left_labels.view(-1),  # (B*16)
                right_labels.view(-1)
            ])  # (2*B*16)
            self.optimizer.zero_grad()
            with autocast():
                case_pred, patch_pred = self.model(left, right)
                case_loss = self.case_criterion(case_pred.squeeze(), case_labels.float())

                patch_loss = self.patch_criterion(
                    patch_pred.view(-1, 2),
                    patch_labels.long().view(-1)
                )
                total_loss = 0.7 * case_loss + 0.3 * patch_loss
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_case_loss += case_loss.item()
            total_patch_loss += patch_loss.item()

        return {
            'case_loss': total_case_loss / len(self.train_loader),
            'patch_loss': total_patch_loss / len(self.train_loader)
        }

    def validate(self):
        self.model.eval()
        case_preds = []
        case_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                left = batch['left_patches'].to(self.device)
                right = batch['right_patches'].to(self.device)
                labels = batch['label'].to(self.device)

                case_pred, _ = self.model(left, right)
                case_preds.append(case_pred.cpu())
                case_labels.append(labels.cpu())

        case_preds = torch.cat(case_preds).numpy()
        case_labels = torch.cat(case_labels).numpy()
        auc = roc_auc_score(case_labels, case_preds)
        return {'auc': auc}

    def train(self, epochs, save_path='best_model.pth'):
        for epoch in range(epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Case Loss: {train_metrics['case_loss']:.4f} | "
                  f"Train Patch Loss: {train_metrics['patch_loss']:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f}")

            if val_metrics['auc'] > self.best_val_auc:
                self.best_val_auc = val_metrics['auc']
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with AUC {self.best_val_auc:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    case_dirs = glob.glob('./data/cls_data/train/case*')

    train_dataset = BATDataset(
        root_paths=case_dirs,
        patch_size=32,
        seg_model_file=None     #'./MG_ATSeg/save_seg_model/seg_best.pth'
    )

    val_dataset = BATDataset(
        root_paths=case_dirs,
        patch_size=32,
        seg_model_file=None     #'./MG_ATSeg/save_seg_model/seg_best.pth'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=5,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    model = CEBATCls()
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(epochs=5)