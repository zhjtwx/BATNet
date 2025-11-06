# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append('./MG_ATSeg')
from model.unet_mam import MAMUnet
from dataset import MyDataset
import mts
from torch.optim import lr_scheduler
import numpy as np
import os
import glob
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# ============================================================
# Training Function
# ============================================================


def trainer(seg_model, train_data, train_epoch, criterion, optimizer):
    """
    Train the segmentation model for one epoch.

    Args:
        seg_model (nn.Module): The segmentation model to be trained.
        train_data (DataLoader): DataLoader providing training batches.
        train_epoch (int): Current training epoch.
        criterion (loss function): Loss function (e.g., BCE+Dice loss).
        optimizer (torch.optim.Optimizer): Optimization algorithm.
    """
    seg_model.train()
    tol_loss = 0
    for idx, batch in enumerate(train_data):
        torch.cuda.empty_cache()
        if use_gpu:
            inputs = Variable(batch[0].cuda())
            labels = Variable(batch[1].cuda())
        else:
            inputs, labels = Variable(batch['0']), Variable(batch['1'])
        optimizer.zero_grad()
        outputs = seg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()
        tol_loss += loss.item()
        t_idx = idx + 1
        if t_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                train_epoch, t_idx, len(train_data), 100. * t_idx / len(train_data), tol_loss / t_idx))



# ============================================================
# Validation Function
# ============================================================


def val(seg_model, val_data, criterion):
    """
    Evaluate model performance on validation or test data.

    Args:
        seg_model (nn.Module): The trained segmentation model.
        val_data (DataLoader): Validation DataLoader.
        criterion (loss function): Loss function.

    Returns:
        float: Average Dice coefficient on the validation set.
    """
    seg_model.eval()
    tol_loss = 0
    tol_dice = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_data):
            torch.cuda.empty_cache()
            if use_gpu:
                inputs = Variable(batch[0].cuda())
                labels = Variable(batch[1].cuda())
            else:
                inputs, labels = Variable(batch['0']), Variable(batch['1'])
            outputs = seg_model(inputs)
            loss = criterion(outputs, labels)
            dice_t = mts.cal_dice(outputs, labels)
            tol_loss += loss.item()
            tol_dice += dice_t.item()
    tol_loss /= len(val_data)
    tol_dice /= len(val_data)
    print('\nVal set: Average loss: {:.6f}\tdice_t: {:.6f}\t\n'.format(tol_loss, tol_dice))
    return tol_dice


def cal_dice(output, target, eps=1e-6):
    output = output.type(torch.cuda.FloatTensor)
    num_t = 2 * (output[:, 1, :, :] * target[:, 1, :, :]).sum()
    den_t = output[:, 1, :, :].sum() + target[:, 1, :, :].sum() + eps
    dice_t = num_t / den_t
    return dice_t


def load_model(lr, epoch_list, num_gpu):
    """
    Initialize model, optimizer, scheduler, and loss function.

    Args:
        lr (float): Initial learning rate.
        epoch_list (list): Milestones for learning rate decay.
        num_gpu (list): List of GPU device IDs for DataParallel.

    Returns:
        tuple: (seg_model, criterion, optimizer, scheduler)
    """
    seg_model = MAMUnet(5, 2)
    if use_gpu:
        seg_model = seg_model.cuda()
        seg_model = nn.DataParallel(seg_model, device_ids=num_gpu)
    criterion = mts.BCEDiceLoss(0.3)
    optimizer = optim.Adam(seg_model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_list, gamma=0.1)
    return seg_model, criterion, optimizer, scheduler


if __name__ == "__main__":

    epochs = 100
    lr = 0.01
    momentum = 0.95
    w_decay = 1e-6
    step_size = 50
    gamma = 0.5
    n_class = 2
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    batch_size = len(num_gpu) * 4
    train_img_paths = glob.glob('./data/seg_data/train/case_001/data_png/*.png')
    train_dataset = MyDataset(train_img_paths)
    val_img_paths = glob.glob('./data/seg_data/train/case_001/data_png/*.png')
    val_dataset = MyDataset(val_img_paths)
    test_img_paths = glob.glob('./data/seg_data/train/case_001/data_png/*.png')
    test_dataset = MyDataset(test_img_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print('mod load...')
    model_dir = "./MG_ATSeg/save_seg_model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    score_dir = os.path.join(model_dir, 'scores')
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    best_dice = 0
    seg_model, criterion, optimizer, scheduler = load_model(lr, [10, 20, 50], num_gpu)

    for epoch in range(epochs):
        print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']}")
        trainer(seg_model, train_loader, epoch, criterion, optimizer)
        scheduler.step()
        val_dice_t = val(seg_model, val_loader, criterion)
        test_dice_t = val(seg_model, test_loader, criterion)
        if val_dice_t > best_dice:
            model_path = os.path.join(model_dir, 'seg_best_demo.pth')
            best_dice = val_dice_t
            torch.save(seg_model, model_path)
