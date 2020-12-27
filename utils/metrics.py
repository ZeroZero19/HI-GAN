import torch
import numpy as np
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from utils.misc import to_numpy

def cal_psnr(clean, noisy, max_val=255, normalized=True):
    """
    Args:
        clean (Tensor): [0, 255], BCHW
        noisy (Tensor): [0, 255], BCHW
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        PSNR per image: (B,)
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)
    mse = F.mse_loss(noisy, clean, reduction='none').view(clean.shape[0], -1).mean(1)
    return 10 * torch.log10(max_val ** 2 / mse)

def pixel_acc(label, pred):
    # _, preds = torch.max(pred, dim=1)
    preds = pred.long()
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def pixel_acc_bce(label, pred):
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (torch.sigmoid(pred).long() == label.long()).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc


def cal_ssim(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)
    ssim = np.array([compare_ssim(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return ssim   


def cal_psnr2(clean, noisy, normalized=True):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)

    psnr = np.array([compare_psnr(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return psnr   


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb



    



