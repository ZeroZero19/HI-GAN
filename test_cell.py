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
from utils.data_loader import load_denoising_test_mix_flyv2, load_denoising_test_mix, fluore_to_tensor
from utils.metrics import cal_psnr, cal_ssim
from utils.misc import mkdirs, stitch_pathes, to_numpy, module_size
from utils.plot import plot_row
import cv2
import os
from shutil import copyfile

plt.switch_backend('agg')


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default='models/cell/hi_gan.pth', type=str, help='the model')
parser.add_argument('--net', type=str, default='all', choices=['N2N', 'DnCNN','UNet_ND', 'UNet_D', 'HI_GAN', 'all'])
parser.add_argument('--batch-size', default=1, type=int, help='test batch size')
parser.add_argument('--data-root', default='testsets/cell', type=str, help='dir to dataset')
parser.add_argument('--out-dir', default='results/cell', type=str, help='dir to dataset')
parser.add_argument('--noise-levels', default=[1, 2, 4, 8, 16], type=str, help='dir to pre-trained model')
parser.add_argument('--image-types', type=str, default='all', choices=['fmd_test_mix', 'our_data', 'all'])
parser.add_argument('--no-cuda', action='store_true', default=False, help='use GPU or not, default using GPU')
parser.add_argument('--save_noi_clean', action='store_true', default=False, help='save_noi_clean')
parser.add_argument('--cuda', type=int, default=4, help='cuda number')
opt = parser.parse_args()


test_batch_size = opt.batch_size
test_seed = 13
cmap = 'inferno'
device = 'cpu' if opt.no_cuda else 'cuda'

noise_levels = opt.noise_levels

if opt.image_types == 'all':
    image_types = ['fmd_test_mix', 'our_data']
else:
    image_types = [opt.image_types]

if opt.net == 'all':
    nets = ['N2N', 'DnCNN','UNet_ND', 'UNet_D', 'HI_GAN', 'all']
else:
    nets = [opt.net]

for net in nets:

    if net == 'N2N':
        model = N2N(1, 1).to(device)
        model.load_state_dict(torch.load('models/cell/n2n.pth'))
    elif net == 'DnCNN':
        model = DnCNN(depth=17,
                      n_channels=64,
                      image_channels=1,
                      use_bnorm=True,
                      kernel_size=3).to(device)
        model.load_state_dict(torch.load('models/cell/dncnn.pth'))
    elif net == 'UNet_ND':
        model = UNet_ND_cell().to(device)
        model = nn.DataParallel(model, list(range(opt.cuda)))
        model.load_state_dict(torch.load('models/cell/unet_nd.pth'))
    elif net == 'UNet_D':
        model = UNet_D_cell().to(device)
        model = nn.DataParallel(model, list(range(opt.cuda)))
        model.load_state_dict(torch.load('models/cell/unet_d.pth'))
    elif net == 'HI_GAN':
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
    four_crop = transforms.Compose([
        transforms.FiveCrop(256),
        transforms.Lambda(lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops[:4]])),
        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])

    for noise_level in noise_levels:
        for image_type in image_types:
            out_dir = os.path.join(opt.out_dir, image_type, net)
            mkdirs(out_dir)
            if image_type == 'fmd_test_mix':
                n_plots = 12
                test_loader = load_denoising_test_mix(opt.data_root,
                                                      batch_size=test_batch_size, noise_levels=[noise_level],
                                                      transform=four_crop, target_transform=four_crop,
                                                      patch_size=256)
            elif image_type == 'our_data':
                n_plots = 12
                test_loader = load_denoising_test_mix_flyv2(opt.data_root,
                                                            batch_size=test_batch_size, noise_levels=[noise_level],
                                                            transform=four_crop, target_transform=four_crop,
                                                            patch_size=256)
            # four crop
            multiplier = 4
            n_test_samples = len(test_loader.dataset) * multiplier

            case = {'noise': noise_level,
                    'type': image_type,
                    'samples': n_test_samples,
                    }
            pprint(case)
            print('Start testing............')

            psnr, ssim = 0., 0.
            psnr_noi, ssim_noi = 0., 0.
            out = {}
            for batch_idx, (noisy, clean, path) in enumerate(test_loader):
                name = os.path.basename(path[0])
                noisy, clean = noisy.to(device), clean.to(device)
                # fuse batch and four crop
                noisy = noisy.view(-1, *noisy.shape[2:])
                clean = clean.view(-1, *clean.shape[2:])
                with torch.no_grad():
                    if net == 'HI_GAN':
                        denoised0 = model0(noisy)
                        denoised1 = model1(noisy)
                        denoised = model(denoised0, denoised1)
                    else:
                        denoised = model(noisy)
                psnr += cal_psnr(clean, denoised).sum().item()
                ssim += cal_ssim(clean, denoised).sum()

                psnr_noi += cal_psnr(clean, noisy).sum().item()
                ssim_noi += cal_ssim(clean, noisy).sum()

            psnr = psnr / n_test_samples
            ssim = ssim / n_test_samples

            psnr_noi = psnr_noi / n_test_samples
            ssim_noi = ssim_noi / n_test_samples
            print('Noisy PSNR/SSIM: %.2f/%.4f'%(psnr_noi,ssim_noi))

            result = {'psnr_dn': '%.2f'%psnr,
                      'ssim_dn': '%.4f'%ssim,}
            case.update(result)
            pprint(result)
            logger.update({f'noise{noise_level}_{image_type}': case})

            with open(out_dir + "/results_{}_{}.txt".format('cpu' if opt.no_cuda else 'gpu', image_type), 'w') as args_file:
                json.dump(logger, args_file, indent=4)



