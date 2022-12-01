from __future__ import print_function, division

from six.moves import zip

import numpy as np

import cv2
import torch
import math
import torch.nn.functional as F

# noinspection PyUnresolvedReferences
from . import _augmentation_cuda

# byte_tensor_base_lut = torch.from_numpy(np.linspace(0, 1, 2 ** 8)).float()
# byte_tensor_contrast_lut = torch.from_numpy(_make_contrast_lut(6, -6, 2 ** 8)).float()
# short_tensor_base_lut = torch.from_numpy(np.linspace(0, 1, 2 ** 16)).float()
# short_tensor_contrast_lut = torch.from_numpy(_make_contrast_lut(6, -6, 2 ** 15)).float()


DTYPE_INFO = {
    'torch.ByteTensor': (
        torch.float32,
        float(-2 ** 8),
        float(2 ** 8 - 1),
    ),
    'torch.cuda.ByteTensor': (
        torch.float32,
        float(-2 ** 8),
        float(2 ** 8 - 1),
    ),
    'torch.ShortTensor': (
        torch.float32,
        float(-2 ** 15),
        float(2 ** 15 - 1),
    ),
    'torch.cuda.ShortTensor': (
        torch.float32,
        float(-2 ** 15),
        float(2 ** 15 - 1),
    ),
    'torch.cuda.FloatTensor': (
        torch.float32,
        -1.0,
        1.0
    ),
    'torch.cuda.DoubleTensor': (
        torch.float64,
        -1.0,
        1.0
    ),
    'torch.cuda.HalfTensor': (
        torch.float32,
        -1.0,
        1.0
    ),
    'torch.FloatTensor': (
        torch.float32,
        -1.0,
        1.0
    ),
    'torch.DoubleTensor': (
        torch.float64,
        -1.0,
        1.0
    ),
    'torch.HalfTensor': (
        torch.float16,
        "egal,",
        1.0
    ),
    'torch.cuda.IntTensor': (
        torch.float32,
        float(-np.iinfo(np.uint32).max - 1),
        float(2 ** 31 - 1)
    ),
    'torch.IntTensor': (
        torch.float32,
        -np.iinfo(np.uint32).max - 1,
        float(2 ** 31 - 1)
    )

}


def add_gamma(
        im_tensor, augs,
        maxv=None,
        in_place=False
):
    """
    A Function that takes a tensor that contains a Batch of Images and a list of dictionaries that contain information about the desired gamma
    values and takes those gamma values to apply gamma correction to the images. This function is Hardware accelerated, so
    be sure that the im_tensor is located on the GPU.

    :param im_tensor: the Tensor that contains the Image data
    :param augs: a list of dictionaries. Each dict should contain a 'color', a 'gamma_gray', a 'gamma_color' and a 'contrast' value
    to specify the behaviour of the gamma augmentation. For further information see :func:`~crumpets.augmentation.randomize_image`
    :param maxv: Maximum value of the entries. This value is data type dependent, so be careful with it. It defaults to "None".
    None indicates that the value is taken according to the data type of the tensor.
    :param in_place: The augmentation can be done in place for performance reasons. It defaults to False, because in place behaviour is
    often not expected.
    """
    if not in_place:
        im_tensor = im_tensor.clone()
    # im_tensor = (im_tensor.double()*1.0/255.0).float()
    maxv = maxv or (DTYPE_INFO[im_tensor.type()])[2]
    if not im_tensor.is_cuda:
        raise Exception("should be cuda Tensor")
    # extract relevant augmentations
    device = im_tensor.device
    colors = [a['color'] if 'color' in a else True for a in augs]  # immer bool?
    gamma_grays = [a['gamma_gray'] if 'gamma_gray' in a else None for a in augs]  # immer 1 float oder None?
    gamma_colors = [a['gamma_color'] if 'gamma_color' in a else None for a in augs]  # immer #c floats oder None?
    contrasts = [a['contrast'] if 'contrast' in a else None for a in augs]  # immer 1 float oder None?

    gamma_grays = torch.tensor([gamma_gray if gamma_gray else 1 for gamma_gray in gamma_grays], dtype=torch.float).to(
        device)
    gamma_colors = torch.tensor(
        [gamma_color if gamma_color and color else im_tensor.shape[1] * [1] for (gamma_color, color) in
         zip(gamma_colors, colors)], dtype=torch.float).to(device)

    contrasts = torch.tensor([contrast if contrast else 0 for contrast in contrasts], dtype=torch.float).to(device)
    N, C, H, W = im_tensor.shape
    _augmentation_cuda.add_gamma(im_tensor, gamma_grays, gamma_colors, contrasts, maxv)
    # im_tensor = (im_tensor.double()*255.0/(1)).byte()

    return im_tensor


def add_noise_rgb(im, augs, minv=None, maxv=None, internal_ftype=None):
    """
    A Function that takes a tensor that contains a Batch of Images and a list of dictionaries that contain information about the desired noise
    and takes this information to add noise according to the that to the images. This noise function tries to mimic the
    rgb noise of a camera sensor, what means that the green value has a lower noise. This function is Hardware accelerated, so
    be sure that the im is located on the GPU.

    :param im: the Tensor that contains the Image data
    :param augs: a list of dictionaries. Each dict should contain a 'noise' value
    to specify the behaviour of the noise augmentation. For further information see :func:`~crumpets.augmentation.randomize_image`
    :param minv: Minimum value of the entries. This value is data type dependent, so be careful with it. It defaults to "None".
    None indicates that the value is taken according to the data type of the tensor.
    :param maxv: Maximum value of the entries. This value is data type dependent, so be careful with it. It defaults to "None".
    None indicates that the value is taken according to the data type of the tensor.
    :param internal_ftype: The type that is used internally to compute the noise. For most types the internal type is float32.
    The type defaults to None, what indicates that a fitting type is chosen according to the input type.
    """
    n, c, h, w = im.shape
    noise_type, minv1, maxv1 = DTYPE_INFO[im.type()]
    noise_type = internal_ftype if internal_ftype else noise_type
    minv = minv if minv else minv1
    maxv = maxv if maxv else maxv1
    strengths = [a['noise'] if 'noise' in a else 0 for a in augs]
    old_im_type = im.dtype
    im = im.type(noise_type)
    s = torch.from_numpy(np.asarray(strengths)[:, None, None, None]).to(noise_type).to(im.device)
    multiplier = torch.tensor([1, 0.5] + [1] * (c - 2), device=im.device, dtype=noise_type)
    # print(s.shape)
    # print(multiplier.shape)
    s = torch.mm(s.reshape(-1, 1), multiplier.reshape(1, -1)).reshape(n, c, 1, 1)
    noisyrb = torch.empty(n, c - 1, h, w, dtype=noise_type, device=im.device).uniform_(minv, maxv).to(im.device)
    noisyg = torch.empty(n, 1, h // 2, w // 2, dtype=noise_type, device=im.device).uniform_(minv, maxv).to(im.device)
    noisyg = torch.nn.functional.interpolate(input=noisyg, size=(h, w), mode="bilinear", align_corners=True)

    # print(noisyrb[:,0].reshape(n,1,h,w).shape)
    # print(noisyg.shape)
    # print(noisyrb[:,1:].reshape(n,c-2,h,w).shape)

    noisy = torch.cat([
        noisyrb[:, 0].reshape(n, 1, h, w),
        noisyg,
        noisyrb[:, 1:].reshape(n, c - 2, h, w)
    ], dim=1)

    noisy = torch.addcmul(im, 1, s, noisy.reshape(n, c, h, w))
    im = noisy.clamp(0, maxv).type(old_im_type)
    return im


# takes the minimum and maximum allowed value of the input image and an internal floating type that is used to add the noise.
# If not given reasonable defaults are assumed. The cuda parameter is obsolete, it is not used anymore. Use the device property of the image tensor instead
def add_noise_other(im, augs, minv=None, maxv=None, internal_ftype=None):
    """
    A Function that takes a tensor that contains a Batch of Images and a list of dictionaries that contain information about the desired noise
    and adds noise according to that to the images. This function is Hardware accelerated, so
    be sure that the im tensor is located on the GPU.

    :param im: the Tensor that contains the Image data
    :param augs: a list of dictionaries. Each dict should contain a 'noise' value
    to specify the behaviour of the noise augmentation. For further information see :func:`~crumpets.augmentation.randomize_image`
    :param minv: Minimum value of the entries. This value is data type dependent, so be careful with it. It defaults to "None".
    None indicates that the value is taken according to the data type of the tensor.
    :param maxv: Maximum value of the entries. This value is data type dependent, so be careful with it. It defaults to "None".
    None indicates that the value is taken according to the data type of the tensor.
    :param internal_ftype: The type that is used internally to compute the noise. For most types the internal type is float32.
    The type defaults to None, what indicates that a fitting type is chosen according to the input type.
    """
    n, c, h, w = im.shape
    noise_type, minv1, maxv1 = DTYPE_INFO[im.type()]
    noise_type = internal_ftype if internal_ftype else noise_type
    minv = minv if minv else minv1
    maxv = maxv if maxv else maxv1
    strengths = [a['noise'] if 'noise' in a else 0 for a in augs]
    old_im_type = im.dtype
    im = im.type(noise_type)
    s = torch.from_numpy(np.asarray(strengths)[:, None, None, None]).to(noise_type).to(im.device)
    noisy = torch.empty(n, c, h, w, dtype=noise_type, device=im.device).uniform_(minv, maxv).to(im.device)
    noisy = torch.addcmul(im, 1, s, noisy)
    im = noisy.clamp(0, maxv).type(old_im_type)
    return im


#
# def add_noise_other_old(im, augs, cuda=True):
#     n, c, h, w = im.shape
#     noise_type, minv, maxv = DTYPE_INFO[im.type()]
#     strengths = [a['noise'] if 'noise' in a else None for a in augs]
#     if not any(strengths):
#         return im
#     s = torch.from_numpy(np.asarray(strengths)[:, None, None, None]).float()
#     if cuda:
#         with torch.cuda.device(im.get_device()):
#             s = s.cuda()
#             s_add = (s * minv).cuda()
#             noisy = torch.cuda.FloatTensor(n, c, h, w).random_(0, maxv - minv)
#     else:
#         s_add = s * minv
#         noisy = torch.FloatTensor(n, c, h, w).random_(0, maxv - minv)
#     noisy *= s
#     noisy += s_add
#     noisy = noisy.int() + im.int()
#     im = noisy.clamp(0, maxv).byte()
#     return im


# use conv2d with groups. conv of (1,N*C,H,W)


# def add_blur_old(im, augs, cuda=True):
#     n, c, h, w = im.shape
#     threshold = 0.0  # 0.1 / 448
#     # extract relevant augmentation
#     sigmas = np.asarray([a['blur'] if 'blur' in a and a['blur'] > threshold else 0.0 for a in augs])
#     if not any(sigmas):
#         return im
#     # sigma is relative to image width
#     sigmas *= im.shape[3]
#
#     # prepare kernels
#     ksizes = [int(s * 6.6 - 2.3) | 1 if s >= 0.2 else 0 for s in sigmas]  # for roughly s <= 0.2 ksize would be < 0
#     ksizes = [k if k >= 3 else 3 for k in ksizes]
#     maxsize = max(ksizes)
#     kernels = [cv2.getGaussianKernel(maxsize, sigma) if all([ksize, sigma]) else None
#                for ksize, sigma in zip(ksizes, sigmas)]
#     kernels = [kernel.T * kernel if kernel is not None else None for kernel in kernels]
#     validkernels = [i for i, k in enumerate(kernels) if k is not None and k.size > 1]
#     if len(validkernels) == 0:
#         return im
#     kernels = [torch.from_numpy(k.astype(np.float32)).repeat(c, 1, 1)
#                for k in kernels if k is not None and k.size > 1]
#
#     kernels = torch.stack(kernels);
#     kernels = kernels.reshape(-1, 1, maxsize, maxsize).cuda()
#     im = im.reshape(1, -1, h, w).float()
#
#     im = F.conv2d(im, kernels, groups = n*c, padding = math.floor(maxsize/2))
#     im = im.byte().reshape(n,c,h,w)
#
#     # for i, k in enumerate(kernels):
#     #     if cuda:
#     #         with torch.cuda.device(im.get_device()):
#     #             k = k.cuda()
#     #     varim = im[validkernels[i], None, :, :, :].float()
#     #     im[validkernels[i], :, :, :] = F.conv2d(varim, k, padding=k.data.shape[3] // 2, groups=c).byte().data
#
#     return im


def add_blur(im, augs):
    """
    A Function that takes a tensor that contains a Batch of Images and a list of dictionaries that contain information about the desired blur
    and takes this information to blur the image. This function is Hardware accelerated, so
    be sure that the im is located on the GPU.

    :param im: the Tensor that contains the Image data
    :param augs: a list of dictionaries. Each dict should contain a 'blur' value. This blur indicates the sigma value of
    the normal distribution filter that is used to blur the image. Also note that the blur value should be relative to
    the image size, to achieve the same optical blur effect on different image sizes.
    to specify the behaviour of the noise augmentation. For further information see :func:`~crumpets.augmentation.randomize_image`

    """
    n, c, h, w = im.shape
    threshold = 0.0  # 0.1 / 448
    # extract relevant augmentation
    imtype = im.type()

    if type(augs) is not list:
        raise Exception("Augmentations should be a list")
    if n != len(augs):
        raise Exception(
            "the number of augmentations should match the batch size, expected: " + str(n) + " but got " + str(
                len(augs)) + " \naugmentations: " + str(augs))
    sigmas = np.asarray([a['blur'] if 'blur' in a and a['blur'] > threshold else 0.0 for a in augs])

    if not any(sigmas):
        return im
    # sigma is relative to image width
    sigmas *= im.shape[3]
    # prepare kernels
    ksizes = [int(s * 6.6 - 2.3) | 1 if s >= 0.2 else 0 for s in sigmas]  # for roughly s <= 0.2 ksize would be < 0
    ksizes = [k if k >= 3 else 3 for k in ksizes]

    maxsize = max(ksizes)
    sigmas = [sigma or 1e-8 for sigma in sigmas]
    kernels = [cv2.getGaussianKernel(maxsize, sigma) for ksize, sigma in zip(ksizes, sigmas)]
    kernels = [torch.from_numpy(k.astype(np.float32)).repeat(c, 1, 1)
               for k in kernels if k is not None and k.size > 1]
    kernels = torch.stack(kernels)

    kernels = kernels.reshape(-1, 1, maxsize, 1).to(im.device)
    im = F.pad(im.float(), (0, 0, math.floor(maxsize / 2), math.floor(maxsize / 2)), mode='replicate')
    _, _, newh, neww = im.shape
    im = im.reshape(1, -1, newh, neww)
    im = F.conv2d(im, kernels, groups=n * c)
    im = F.pad(im.float(), (math.floor(maxsize / 2), math.floor(maxsize / 2), 0, 0), mode='replicate')
    _, _, newh, neww = im.shape
    im = im.reshape(1, -1, newh, neww)
    im = F.conv2d(im, kernels.reshape(-1, 1, 1, maxsize), groups=n * c)
    im = im.type(imtype).reshape(n, c, h, w)

    # for i, k in enumerate(kernels):
    #     if cuda:
    #         with torch.cuda.device(im.get_device()):
    #             k = k.cuda()
    #     varim = im[validkernels[i], None, :, :, :].float()
    #     im[validkernels[i], :, :, :] = F.conv2d(varim, k, padding=k.data.shape[3] // 2, groups=c).byte().data

    return im
