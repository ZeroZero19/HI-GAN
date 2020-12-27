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
unetnd = UNet_ND().to(device)
unetd = UNet_D().to(device)
higan = HI_GAN().to(device)

# Load models
if opt.JPEG:
    unetnd_path = 'models/unet_nd_jpeg.pth'
    unetd_path = 'models/unet_d_jpeg.pth'
    higan_path = 'models/hi_gan_jpeg.pth'
else:
    unetnd_path = 'models/unet_nd.pth'
    unetd_path = 'models/unet_d.pth'
    higan_path = 'models/hi_gan.pth'

if (device.type == 'cuda') and (ngpu >= 1):
    unetnd = nn.DataParallel(unetnd, list(range(ngpu)))
    unetd = nn.DataParallel(unetd, list(range(ngpu)))
    higan = nn.DataParallel(higan, list(range(ngpu)))

unetnd.load_state_dict(torch.load(unetnd_path), strict=False)
unetd.load_state_dict(torch.load(unetd_path), strict=False)
higan.load_state_dict(torch.load(higan_path))

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
        gt_dn = unetnd(imorig)
        gf_dn = unetd(imorig)
        higan_dn = higan(gf_dn, gt_dn)

    # save by save_image
    save_img_dir = os.path.join(opt.out, img_folder)
    # create result folder
    try:
        os.makedirs(os.path.join(opt.out, img_folder))
    except OSError:
        pass

    if opt.JPEG:
        save_image(make_grid(higan_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_HIGAN_JPEG_denoi.png' % (save_img_dir, img_name))
    else:
        save_image(make_grid(higan_dn.clamp(0., 1.), nrow=8, normalize=False, scale_each=False),
                   '%s/%s_HIGAN_denoi.png' % (save_img_dir, img_name))
