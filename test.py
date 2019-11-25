#!/usr/bin/env python

import os
import glob
import cv2
import argparse
import numpy as np

from models import *

from torchvision.utils import save_image, make_grid

parser = argparse.ArgumentParser()
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--inp', type=str, default='testsets/DND_20_rand_patches', help='input folder')
parser.add_argument('--out', type=str, default='results', help='output folder')
parser.add_argument('--JPEG', action='store_true', help="for JPEG images")

opt = parser.parse_args()
print(opt)

# Number of GPUs available. Use 0 for CPU mode.
ngpu = opt.nGPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Create models
unet_nd = GNet().to(device)
unet_d = GNet().to(device)
boostnet = BoostNet().to(device)

# Load models
if opt.JPEG:
    unet_nd_path = 'models/UNet-ND_JPEG.pth'
    unet_d_path = 'models/UNet-D_JPEG.pth'
else:
    unet_nd_path = 'models/UNet-ND.pth'
    unet_d_path = 'models/UNet-D.pth'

Boostnet_path = 'models/Boost-Net.pth'
if (device.type == 'cuda') and (ngpu >= 1):
    unet_nd = nn.DataParallel(unet_nd, list(range(ngpu)))
    unet_d = nn.DataParallel(unet_d, list(range(ngpu)))
    boostnet = nn.DataParallel(boostnet, list(range(ngpu)))

unet_nd.load_state_dict(torch.load(unet_nd_path), strict=False)
unet_d.load_state_dict(torch.load(unet_d_path), strict=False)
boostnet.load_state_dict(torch.load(Boostnet_path))

# Denoise
print('\n> Test set', opt.inp)
files = []
types = ('*.bmp', '*.png', '*.jpg', '*.JPEG', '*.tif')

for tp in types:
    files.extend(glob.glob(os.path.join(opt.inp, tp)))
files.sort()

for i, item in enumerate(files):
    torch.cuda.empty_cache()
    print("\tfile: %s" % item)
    img_folder = os.path.basename(os.path.dirname(item))
    img_name = os.path.basename(item)
    img_name = os.path.splitext(img_name)[0]

    # Read img
    imorig = cv2.imread(item)
    imorig = imorig[:, :, ::-1] / 255.0
    imorig = np.array(imorig).astype('float32')

    imorig = np.expand_dims(imorig.transpose(2, 0, 1), 0)
    imorig = torch.Tensor(imorig).to(device)

    with torch.no_grad():
        unet_nd_dn = unet_nd(imorig)
        unet_d_dn = unet_d(imorig)
        boost_dn = boostnet(unet_d_dn, unet_nd_dn)

    # save by save_image
    save_img_dir = os.path.join(opt.out, img_folder)
    # create result folder
    try:
        os.makedirs(os.path.join(opt.out, img_folder))
    except OSError:
        pass

    if opt.JPEG:
        save_image(make_grid(unet_nd_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_UNet-ND_JPEG.png' % (save_img_dir, img_name))
        save_image(make_grid(unet_d_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_UNet-D_JPEG.png' % (save_img_dir, img_name))
        save_image(make_grid(boost_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_Boost-Net_JPEG.png' % (save_img_dir, img_name))
    else:
        save_image(make_grid(unet_nd_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_UNet-ND.png' % (save_img_dir, img_name))
        save_image(make_grid(unet_d_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_UNet-D.png' % (save_img_dir, img_name))
        save_image(make_grid(boost_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_Boost-Net.png' % (save_img_dir, img_name))
