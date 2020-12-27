import json
import os
from time import time

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms.functional import _is_pil_image

__all__ = ['fluore_to_tensor', 'DenoisingFolder',
           'DenoisingTestMixFolder', 'load_denoising','load_denoising_test_mix']

IMG_EXTENSIONS = ['.png']


def encode_mask(file, clean):
    scale = 10
    if 'Confocal_BPAE_B' in file:
        mask_score = scale*2
    if 'Confocal_BPAE_G' in file:
        mask_score = scale*4
    if 'Confocal_BPAE_R' in file:
        mask_score = scale*6
    if 'Confocal_FISH' in file:
        mask_score = scale*8
    if 'Confocal_MICE' in file:
        mask_score = scale*10
    if 'TwoPhoton_BPAE_B' in file:
        mask_score = scale*12
    if 'TwoPhoton_BPAE_G' in file:
        mask_score = scale*14
    if 'TwoPhoton_BPAE_R' in file:
        mask_score = scale*16
    if 'TwoPhoton_MICE' in file:
        mask_score = scale*18
    if 'WideField_BPAE_B' in file:
        mask_score = scale*20
    if 'WideField_BPAE_G' in file:
        mask_score = scale*22
    if 'WideField_BPAE_R' in file:
        mask_score = scale*24
    if 'Green' in file:
        mask_score = scale*7
    if 'Red' in file:
        mask_score = scale*14
    if 'Magenta' in file:
        mask_score = scale*21

    mask = clean > threshold_otsu(clean.numpy(), nbins=2)
    mask = mask.float()* mask_score
    mask = mask.div(255).sub(0.5)

    return torch.tensor(mask)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, tuple(IMG_EXTENSIONS))


def pil_loader(path):
    img = Image.open(path)
    return img


def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.

    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not (_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)

    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        # RBG to grayscale:
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img




class DenoisingTestMixFolder(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, noise_levels, transform, target_transform):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]

        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels

        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_mix',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'FMD_test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')

        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            for fname in sorted(os.listdir(noise_dir)):
                # 'TwoPhoton_BPAE_B','TwoPhoton_MICE', 'Confocal_MICE','Confocal_BPAE_B','WideField_BPAE_B',
                # if is_image_file(fname) and ('_BPAE_B' in fname or '_MICE' in fname):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    samples.append((noisy_file, clean_file))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        name = os.path.basename(self.samples[index][0])

        return noisy, clean, name

    def __len__(self):
        return len(self.samples)

def load_denoising_test_mix(root, batch_size, noise_levels, loader=pil_loader,
                            transform=None, target_transform=None, patch_size=256):
    """
    files: root/test_mix/noise_level/captures.png

    Args:
        root (str):
        batch_size (int):
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            fluore_to_tensor,
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
    # the same
    target_transform = transform
    dataset = DenoisingTestMixFolder(root, loader, noise_levels, transform,
                                     target_transform)
    kwargs = {'num_workers': 0, 'pin_memory': False} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, **kwargs)

    return data_loader

class DenoisingTestMixFolder_flyv2(torch.utils.data.Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, noise_levels, transform, target_transform):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]

        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels

        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()

        dataset_info = {'Dataset': 'test_flyv2',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'our_data')
        gt_dir = os.path.join(test_mix_dir, 'gt')

        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            for fname in sorted(os.listdir(noise_dir)):
                # 'TwoPhoton_BPAE_B','TwoPhoton_MICE', 'Confocal_MICE','Confocal_BPAE_B','WideField_BPAE_B',
                # if is_image_file(fname) and ('_BPAE_B' in fname or '_MICE' in fname):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname.replace(fname.split('_')[-1],'Average.png'))
                    samples.append((noisy_file, clean_file))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)

        name = os.path.basename(self.samples[index][0])

        return noisy, clean, name

    def __len__(self):
        return len(self.samples)

def load_denoising_test_mix_flyv2(root, batch_size, noise_levels, loader=pil_loader,
                            transform=None, target_transform=None, patch_size=256):
    """
    files: root/test_mix/noise_level/captures.png

    Args:
        root (str):
        batch_size (int):
        noise_levels (seq): e.g. [1, 2, 4], or [1, 2, 4, 8]
        types (seq, None): e.g.     [`microscopy_cell`]
        transform (torchvision.transform): transform to noisy images
        target_transform (torchvision.transform): transforms to clean images
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            fluore_to_tensor,
            transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
        ])
    # the same
    target_transform = transform
    dataset = DenoisingTestMixFolder_flyv2(root, loader, noise_levels, transform,
                                     target_transform)
    kwargs = {'num_workers': 0, 'pin_memory': False} \
        if torch.cuda.is_available() else {}
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, **kwargs)

    return data_loader





if __name__ == '__main__':
    root = 'path/to/denoising/dataset'
    loader = pil_loader
    train = True
    noise_levels = [1, 2, 4]
    types = ['TwoPhoton_MICE']
    # types = None
    captures = 10
    patch_size = 128
    batch_size = 16
    transform = None
    target_transform = None
    test_fov = 19
    tic = time()
    # dataset = DenoisingFolder(root, train, noise_levels, types=types, test_fov=19,
    #     captures=2, transform=None, target_transform=None, loader=pil_loader)
    # print(time()-tic)
    # print(dataset.samples[0])
    # print(dataset[0])
    kwargs = {'drop_last': False}
    add_kwargs = {'num_workers': 4, 'pin_memory': True} \
        if torch.cuda.is_available() else {}
    kwargs.update(add_kwargs)

    loader = load_denoising(root, train, batch_size, noise_levels=noise_levels,
                            types=types, captures=captures, patch_size=patch_size, transform=transform,
                            target_transform=target_transform, loader=pil_loader, test_fov=test_fov)

    for batch_size, (noisy, clean) in enumerate(loader):
        print(noisy.shape)
        print(clean.shape)
        break

    transform = transforms.Compose([
        transforms.FiveCrop(patch_size),
        transforms.Lambda(lambda crops: torch.stack([
            fluore_to_tensor(crop) for crop in crops])),
        transforms.Lambda(lambda x: x.float().div(255).sub(0.5))
    ])

    test_loader = load_denoising_test_mix(root, batch_size=32, noise_levels=noise_levels,
                                          loader=pil_loader, transform=transform, patch_size=patch_size)

    print(len(test_loader.dataset))
    print(test_loader.dataset.samples[0])

    for batch_size, (noisy, clean) in enumerate(test_loader):
        print(noisy.shape)
        print(clean.shape)
        break

    train_loader = load_denoising(root, batch_size, noise_levels, types=None,
                                            patch_size=256, transform=transform, target_transform=transform,
                                            loader=pil_loader,
                                            test_fov=19)

    for batch_size, (noisy_input, noisy_target, clean) in enumerate(train_loader):
        print(noisy_input.shape)
        print(noisy_target.shape)
        print(clean.shape)
        break
