import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import nibabel as nib
try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

def load_nii(load_fp):
    im = nib.load(str(load_fp))
    try:
        return np.asanyarray(im.dataobj), np.asanyarray(im.affine)
    except:
        return np.asanyarray(im.dataobj)


def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets, eps=0.0000001):
        class_num = logits.size(1)
        dice_sum = 0
        for i in range(1, class_num):
            inter = torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :])
            dice = (2. * inter + eps) / (union + eps)
            dice_sum += dice
        return 1 - dice_sum / (class_num - 1)


class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()
    def forward(self, logits, targets):
        class_num = logits.size(1)
        num_sum = torch.sum(targets, dim=(0, 2, 3))

        w = torch.Tensor([1.0, 1.0]).cuda()
        for i in range(2):
            w[i] = (torch.sum(num_sum) + 0.00000001) / (num_sum[i] + 0.00000001)
        flag = torch.sum(w)
        for i in range(2):
            w[i] = w[i]/flag
        dice_sum = 0

        for i in range(2):
            inter = torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = (torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :]))
            dice = w[i]*(2. * inter + 0.00000001) / (union + 0.00000001)
            dice_sum += dice
        return 1 - dice_sum/class_num

class WeightDiceLoss_map(nn.Module):
    def __init__(self):
        super(WeightDiceLoss_map, self).__init__()
    def forward(self, logits, targets, map ):
        class_num = logits.size(1)
        num_sum = torch.sum(targets, dim=(0, 2, 3))

        w = torch.Tensor([1.0, 1.0]).cuda()
        for i in range(2):
            w[i] =(torch.sum(num_sum) + 1) / (num_sum[i] + 1)
        flag = torch.sum(w)
        for i in range(2):
            w[i] = w[i]/flag
        dice_sum = 0
        for i in range(2):
            inter = map * torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
            union = (torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :]))
            dice = w[i]*(2. * inter + 1) / (union + 1)
            dice_sum += dice
                #
                # inter = map * torch.sum(logits[:, i, :, :] * targets[:, i, :, :])
                # union = (torch.sum(logits[:, i, :, :]) + torch.sum(targets[:, i, :, :]))
                # dice = w[i]*(2. * inter + 1) / (union + 1)
                # dice_sum += dice
        print('__________', dice_sum)
        return 1 - dice_sum/class_num

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


class BCEDiceLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        dice = DiceMeanLoss()(input, target)
        return self.weight * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def cal_dice(output, target, eps=1e-6):
    output = output.type(torch.cuda.FloatTensor)
    num_t = 2 * (output[:, 1, :, :] * target[:, 1, :, :]).sum()
    den_t = output[:, 1, :, :].sum() + target[:, 1, :, :].sum() + eps
    dice_t = num_t / den_t
    return dice_t


def T(logits, targets):
    return torch.sum(targets[:, 2, :, :, :])


def P(logits, targets):
    return torch.sum(logits[:, 2, :, :, :])


def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :, :] * logits[:, 2, :, :, :])

