import argparse
import json
from argparse import Namespace
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

import torch
import torch.nn as nn
from models import UNet_D_cell, UNet_ND_cell, HI_GAN_cell
from model_n2n import N2N
from model_dncnn import DnCNN
from utils.data_loader import load_denoising_test_mix_flyv2, load_denoising_test_mix, fluore_to_tensor, pil_loader
from utils.metrics import cal_psnr, cal_ssim
from utils.misc import mkdirs, stitch_pathes, to_numpy, module_size
# from skimage.measure import compare_psnr, compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils.plot import plot_row
import cv2
import os

import tkinter
import matplotlib

matplotlib.use('TkAgg')

plt.switch_backend('agg')


parser = argparse.ArgumentParser()
# parser.add_argument('-m', '--model', default='models/cell/hi_gan.pth', type=str, help='the model')
parser.add_argument('--net', type=str, default='HI_GAN', choices=['N2N', 'DnCNN','UNet_ND', 'UNet_D', 'HI_GAN'])
parser.add_argument('--batch-size', default=1, type=int, help='test batch size')
parser.add_argument('--inp-dir', default='testsets/cell/demo/avg2/Confocal_BPAE_B_4.png', type=str, help='dir to dataset')
parser.add_argument('--out-dir', default='results/cell/demo', type=str, help='dir to dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
parser.add_argument('--save_img', action='store_true', default=True, help='save_noi_clean')
parser.add_argument('--ground-truth', action='store_false', default=True, help='has clean img')
parser.add_argument('--gray', action='store_true', default=False, help='has rgb img')
parser.add_argument('--cuda', type=int, default=4, help='cuda number')
opt = parser.parse_args()


test_batch_size = opt.batch_size
cmap = 'inferno'
device = 'cpu' if opt.no_cuda else 'cuda'

out_dir = opt.out_dir
mkdirs(out_dir)


if opt.net == 'N2N':
    model = N2N(1, 1).to(device)
    model.load_state_dict(torch.load('models/cell/n2n.pth'))
elif opt.net == 'DnCNN':
    model = DnCNN(depth=17,
                  n_channels=64,
                  image_channels=1,
                  use_bnorm=True,
                  kernel_size=3).to(device)
    model.load_state_dict(torch.load('models/cell/dncnn.pth'))
elif opt.net == 'UNet_ND':
    model = UNet_ND_cell().to(device)
    model = nn.DataParallel(model, list(range(opt.cuda)))
    model.load_state_dict(torch.load('models/cell/unet_nd.pth'))
elif opt.net == 'UNet_D':
    model = UNet_D_cell().to(device)
    model = nn.DataParallel(model, list(range(opt.cuda)))
    model.load_state_dict(torch.load('models/cell/unet_d.pth'))
elif opt.net == 'HI_GAN':
    model0 = UNet_ND_cell().to(device)
    model1 = UNet_D_cell().to(device)
    model0 = nn.DataParallel(model0, list(range(opt.cuda)))
    model1 = nn.DataParallel(model1, list(range(opt.cuda)))
    model0.load_state_dict(torch.load('models/cell/unet_nd.pth'))
    model1.load_state_dict(torch.load('models/cell/unet_d.pth'))
    model0.eval()
    model1.eval()

    model = HI_GAN_cell().to(device)

    model = nn.DataParallel(model, list(range(opt.cuda)))
    model.load_state_dict(torch.load('models/cell/hi_gan.pth'))

model.eval()

def convert(x):
    x = (x.transpose(1,2,0)+0.5)*255
    return x.clip(0,255).astype('uint8')


logger = {}
multiplier = 4
four_crop = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(lambda crops: torch.stack([
        fluore_to_tensor(crop) for crop in crops[:4]])),
    transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
    ])

if not opt.gray:

    if 'Red_' in opt.inp_dir or 'Green_' in opt.inp_dir or 'Magenta_' in opt.inp_dir:
        if 'Red_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir
            noisy_g_path = opt.inp_dir.replace('Red_', 'Green_')
            noisy_b_path = opt.inp_dir.replace('Red_', 'Magenta_')
        elif 'Green_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir.replace('Green_', 'Red_')
            noisy_g_path = opt.inp_dir
            noisy_b_path = opt.inp_dir.replace('Green_', 'Magenta_')
        elif 'Magenta_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir.replace('Magenta_', 'Red_')
            noisy_g_path = opt.inp_dir.replace('Magenta_', 'Green_')
            noisy_b_path = opt.inp_dir
        else:
            print('Error: Please use RGB image or set gray mode')
        if opt.ground_truth:
            lv = noisy_r_path.split('/')[-2]
            avg = noisy_r_path.split('_')[-1]
            clean_r = four_crop(pil_loader(noisy_r_path.replace(lv, 'gt').replace(avg, 'Average.png'))).to(device)
            clean_g = four_crop(pil_loader(noisy_g_path.replace(lv, 'gt').replace(avg, 'Average.png'))).to(device)
            clean_b = four_crop(pil_loader(noisy_b_path.replace(lv, 'gt').replace(avg, 'Average.png'))).to(device)
    else:
        if '_R_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir
            noisy_g_path = opt.inp_dir.replace('_R_', '_G_')
            noisy_b_path = opt.inp_dir.replace('_R_', '_B_')
        elif '_G_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir.replace('_G_', '_R_')
            noisy_g_path = opt.inp_dir
            noisy_b_path = opt.inp_dir.replace('_G_', '_B_')
        elif '_B_' in opt.inp_dir:
            noisy_r_path = opt.inp_dir.replace('_B_', '_R_')
            noisy_g_path = opt.inp_dir.replace('_B_', '_G_')
            noisy_b_path = opt.inp_dir
        else:
            print('Error: Please use RGB image or set gray mode')
        if opt.ground_truth:
            lv = noisy_r_path.split('/')[-2]
            clean_r = four_crop(pil_loader(noisy_r_path.replace(lv, 'gt'))).to(device)
            clean_g = four_crop(pil_loader(noisy_g_path.replace(lv, 'gt'))).to(device)
            clean_b = four_crop(pil_loader(noisy_b_path.replace(lv, 'gt'))).to(device)

    noisy_r = four_crop(pil_loader(noisy_r_path)).to(device)
    noisy_g = four_crop(pil_loader(noisy_g_path)).to(device)
    noisy_b = four_crop(pil_loader(noisy_b_path)).to(device)
    with torch.no_grad():
        if opt.net == 'HI_GAN':
            denoised_r = model(model0(noisy_r), model1(noisy_r))
            denoised_g = model(model0(noisy_g), model1(noisy_g))
            denoised_b = model(model0(noisy_b), model1(noisy_b))
        else:
            denoised_r = model(noisy_r)
            denoised_g = model(noisy_g)
            denoised_b = model(noisy_b)

    noisy_rgb = np.stack([convert(stitch_pathes(to_numpy(noisy_b)))[:, :, 0],
                          convert(stitch_pathes(to_numpy(noisy_g)))[:, :, 0],
                          convert(stitch_pathes(to_numpy(noisy_r)))[:, :, 0]]).transpose(1, 2, 0)
    denoised_rgb =np.stack([convert(stitch_pathes(to_numpy(denoised_b)))[:, :, 0],
                          convert(stitch_pathes(to_numpy(denoised_g)))[:, :, 0],
                          convert(stitch_pathes(to_numpy(denoised_r)))[:, :, 0]]).transpose(1, 2, 0)
    if opt.ground_truth:
        clean_rgb = np.stack([convert(stitch_pathes(to_numpy(clean_b)))[:, :, 0],
                            convert(stitch_pathes(to_numpy(clean_g)))[:, :, 0],
                            convert(stitch_pathes(to_numpy(clean_r)))[:, :, 0]]).transpose(1, 2, 0)

        psnr_noi = compare_psnr(clean_rgb, noisy_rgb)
        ssim_noi = compare_ssim(clean_rgb, noisy_rgb, multichannel=True)

        psnr_dn = compare_psnr(clean_rgb, denoised_rgb)
        ssim_dn = compare_ssim(clean_rgb, denoised_rgb, multichannel=True)

        print('PSNR[noi/dn]: %.2f/%.2f,     SSIM[noi/dn]: %.4f/%.4f'%(psnr_noi, psnr_dn, ssim_noi,ssim_dn))
        # plot
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(1, 3, figsize=(5.4 * 3, 5))
        fig.suptitle('[%s] %s'%(opt.net,os.path.basename(noisy_r_path.replace("_R_","_RGB_"))),fontsize=18)
        axs[0].imshow(noisy_rgb)
        axs[0].set_title("Noisy Image (%.2f/%.4f)"%(psnr_noi,ssim_noi),y=-0.1)
        axs[0].set_axis_off()
        axs[1].imshow(denoised_rgb)
        axs[1].set_title("Denoised Image (%.2f/%.4f)"%(psnr_dn,ssim_dn),y=-0.1)
        axs[1].set_axis_off()
        axs[2].imshow(clean_rgb)
        axs[2].set_title(r'Clean Image ($\infty$/$\infty$)',y=-0.1)
        axs[2].set_axis_off()
    else:
        # plot
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(1, 2, figsize=(5.4 * 2, 5))
        fig.suptitle('[%s] %s'%(opt.net,os.path.basename(noisy_r_path.replace("_R_","_RGB_"))),fontsize=18)
        axs[0].imshow(noisy_rgb)
        axs[0].set_title("Noisy Image",y=-0.1)
        axs[0].set_axis_off()
        axs[1].imshow(denoised_rgb)
        axs[1].set_title("Denoised Image",y=-0.1)
        axs[1].set_axis_off()
    fig.savefig(out_dir + f'/{opt.net}_{os.path.basename(noisy_r_path.replace("_R_", "_RGB_"))}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

else:
    noisy = four_crop(pil_loader(opt.inp_dir)).to(device)

    with torch.no_grad():
        if opt.net == 'HI_GAN':
            denoised0 = model0(noisy)
            denoised1 = model1(noisy)
            denoised = model(denoised0, denoised1)
        else:
            denoised = model(noisy)

    if opt.ground_truth:
        lv = opt.inp_dir.split('/')[-2]

        if 'Red_' in opt.inp_dir or 'Green_' in opt.inp_dir or 'Magenta_' in opt.inp_dir:
            avg = opt.inp_dir.split('_')[-1]
            clean = four_crop(pil_loader(opt.inp_dir.replace(lv, 'gt').replace(avg, 'Average.png'))).to(device)
        else:
            clean = four_crop(pil_loader(opt.inp_dir.replace(lv, 'gt'))).to(device)
        psnr_noi = cal_psnr(clean, noisy).sum().item() / multiplier
        ssim_noi = cal_ssim(clean, noisy).sum() / multiplier

        psnr_dn = cal_psnr(clean, denoised).sum().item() / multiplier
        ssim_dn = cal_ssim(clean, denoised).sum() / multiplier
        print('PSNR[noi/dn]: %.2f/%.2f,     SSIM[noi/dn]: %.4f/%.4f' % (psnr_noi, psnr_dn, ssim_noi, ssim_dn))
        # plot
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(1, 3, figsize=(5.4 * 3, 5))
        fig.suptitle('[%s] %s'%(opt.net,os.path.basename(opt.inp_dir)),fontsize=18)
        axs[0].imshow(convert(stitch_pathes(to_numpy(noisy)))[:, :, 0],cmap=cmap)
        axs[0].set_title("Noisy Image (%.2f/%.4f)" % (psnr_noi, ssim_noi), y=-0.1)
        axs[0].set_axis_off()
        axs[1].imshow(convert(stitch_pathes(to_numpy(denoised)))[:, :, 0],cmap=cmap)
        axs[1].set_title("Denoised Image (%.2f/%.4f)" % (psnr_dn, ssim_dn), y=-0.1)
        axs[1].set_axis_off()
        axs[2].imshow(convert(stitch_pathes(to_numpy(clean)))[:, :, 0],cmap=cmap)
        axs[2].set_title(r'Clean Image ($\infty$/$\infty$)', y=-0.1)
        axs[2].set_axis_off()
    else:
        # plot
        matplotlib.use('TkAgg')
        fig, axs = plt.subplots(1, 2, figsize=(5.4 * 2, 5))
        fig.suptitle('[%s] %s'%(opt.net,os.path.basename(opt.inp_dir)),fontsize=18)
        axs[0].imshow(convert(stitch_pathes(to_numpy(noisy)))[:, :, 0],cmap=cmap)
        axs[0].set_title("Noisy Image", y=-0.1)
        axs[0].set_axis_off()
        axs[1].imshow(convert(stitch_pathes(to_numpy(denoised)))[:, :, 0],cmap=cmap)
        axs[1].set_title("Denoised Image", y=-0.1)
        axs[1].set_axis_off()

    fig.savefig(out_dir + f'/{opt.net}_{os.path.basename(opt.inp_dir)}', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)