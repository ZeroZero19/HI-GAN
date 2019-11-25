import numpy as np
import cv2
from skimage.measure.simple_metrics import compare_psnr
from skimage.measure import compare_ssim
import logging
import random

def load_nlf(info, img_id):
    nlf = {}
    nlf_h5 = info[info["nlf"][0][img_id]]
    nlf["a"] = nlf_h5["a"][0][0]
    nlf["b"] = nlf_h5["b"][0][0]
    return nlf

def load_sigma_srgb(info, img_id, bb):
    nlf_h5 = info[info["sigma_srgb"][0][img_id]]
    sigma = nlf_h5[0,bb]
    return sigma

def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)

def variable_to_cv2_image(varim):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res

def batch_psnr(img, imclean, data_range):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
    return psnr/img_cpu.shape[0]

def batch_ssim(img, imclean, data_range):
    r"""
    Computes the SSIM along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    ssim = 0
    for i in range(img_cpu.shape[0]):
        ssim += compare_ssim(imgclean[i, :, :, :].transpose(1, 2, 0),
                             img_cpu[i, :, :, :].transpose(1, 2, 0),
                             data_range=data_range,
                             multichannel=True)
    return ssim/img_cpu.shape[0]

def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    # out = np.transpose(image, (1, 2, 0))
    out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    # return np.transpose(out, (2, 0, 1))
    return out

def DATAaugmentation(imrefer, imnoisy, imfull, imcbdn, mode, crop=[50,50], train=None):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    assert imrefer.shape[1] >= crop[0]
    assert imrefer.shape[2] >= crop[1]
    if train is not None:
        x = random.randint(0, imrefer.shape[1] - crop[0])
        y = random.randint(0, imrefer.shape[2] - crop[0])
    else:
        x = int((imrefer.shape[1] - crop[0])/2)
        y = int((imrefer.shape[2] - crop[1])/2)
    refer = np.transpose(imrefer[:, x:x+crop[0], y:y+crop[1]], (1, 2, 0))
    noisy = np.transpose(imnoisy[:, x:x+crop[0], y:y+crop[1]], (1, 2, 0))
    full = np.transpose(imfull[:, x:x+crop[0], y:y+crop[1]], (1, 2, 0))
    cbdn = np.transpose(imcbdn[:, x:x+crop[0], y:y+crop[1]], (1, 2, 0))

    # plt.imshow(refer)
    # plt.show()
    if mode == 0:
        # original
        refer = refer
        noisy = full
        full = noisy
        cbdn = cbdn
    elif mode == 1:
        # flip up and down
        refer = np.flipud(refer)
        noisy = np.flipud(noisy)
        full = np.flipud(full)
        cbdn = np.flipud(cbdn)
    elif mode == 2:
        # rotate counterwise 90 degree
        refer = np.rot90(refer)
        noisy = np.rot90(noisy)
        full = np.rot90(full)
        cbdn = np.rot90(cbdn)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        refer = np.rot90(refer)
        noisy = np.rot90(noisy)
        full = np.rot90(full)
        cbdn = np.rot90(cbdn)
        refer = np.flipud(refer)
        noisy = np.flipud(noisy)
        full = np.flipud(full)
        cbdn = np.flipud(cbdn)
    elif mode == 4:
        # rotate 180 degree
        refer = np.rot90(refer, k=2)
        noisy = np.rot90(noisy, k=2)
        full = np.rot90(full, k=2)
        cbdn = np.rot90(cbdn, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        refer = np.rot90(refer, k=2)
        noisy = np.rot90(noisy, k=2)
        full = np.rot90(full, k=2)
        cbdn = np.rot90(cbdn, k=2)
        refer = np.flipud(refer)
        noisy = np.flipud(noisy)
        full = np.flipud(full)
        cbdn = np.flipud(cbdn)
    elif mode == 6:
        # rotate 270 degree
        refer = np.rot90(refer, k=3)
        noisy = np.rot90(noisy, k=3)
        full = np.rot90(full, k=3)
        cbdn = np.rot90(cbdn, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        refer = np.rot90(refer, k=3)
        noisy = np.rot90(noisy, k=3)
        full = np.rot90(full, k=3)
        cbdn = np.rot90(cbdn, k=3)
        refer = np.flipud(refer)
        noisy = np.flipud(noisy)
        full = np.flipud(full)
        cbdn = np.flipud(cbdn)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(refer, (2, 0, 1)), np.transpose(noisy, (2, 0, 1)),\
           np.transpose(full, (2, 0, 1)), np.transpose(cbdn, (2, 0, 1))

def init_logger_data(data_dir):
    r"""Initializes a logging.Logger in order to log the data

    Args:
        data_dir: path to the folder with the data
    """
    from os.path import join

    logger = logging.getLogger('datalog')
    logger.setLevel(level=logging.INFO)
    fh = logging.FileHandler(join(data_dir, 'log.txt'), mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger