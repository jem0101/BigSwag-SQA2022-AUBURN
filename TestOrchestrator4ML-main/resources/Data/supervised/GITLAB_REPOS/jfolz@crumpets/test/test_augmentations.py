import os
import os.path as pt
import sys

ROOT = pt.dirname(__file__)
parent = pt.abspath(pt.join(ROOT, os.pardir))

for p in (os.pardir, parent):
    try:
        sys.path.remove(p)
    except ValueError:
        pass
print(sys.path)

import math

import numpy as np
import torch
import crumpets.torch.augmentation_cuda as cudaaugs

import crumpets.augmentation_cpu as cpuaugs


def test_noise_rgb_semantically_float():
    torch.manual_seed(1337)
    for strength in np.arange(0, 0.55, 0.05):
        N, C, H, W = (2, 3, 3000, 3000)
        im = (torch.zeros(N, C, H, W) + 0.5)
        im = im.cuda()
        expected_std = math.sqrt((2 * strength) ** 2 / 12)
        expected_mean = 0.5

        a = {'noise': strength}
        augs = [a] * N
        noisy = cudaaugs.add_noise_rgb(im, augs)
        mean = noisy.mean()
        std = noisy.transpose(0, 1).contiguous().view(C, -1).std(dim=1, unbiased=False)
        assert abs(std[0] - expected_std) <= 0.001
        assert abs(mean - expected_mean) <= 0.0001


def test_noise_rgb_semantically_byte():
    for strength in np.arange(0, 0.55, 0.05):
        N, C, H, W = (2, 3, 3000, 3000)
        im = (torch.zeros(N, C, H, W)).byte() + 2 ** 7
        im = im.cuda()
        expected_std = math.sqrt((2 * strength * (2 ** 8 - 1)) ** 2 / 12)
        expected_mean = 2 ** 7
        a = {'noise': strength}
        augs = [a] * N
        noisy = cudaaugs.add_noise_rgb(im, augs)
        mean = noisy.sum().float() / (N * C * H * W)
        std = ((noisy.float() - mean).pow(2).transpose(0, 1).contiguous().view(C, -1).sum(dim=1) / (N * H * W)).sqrt()
        # std = noisy.transpose(0,1).reshape(C,-1).std(dim=1, unbiased = False)
        assert abs(std[0] - expected_std) / (2 ** 8 - 1) <= 0.01
        assert abs(mean - expected_mean) / (2 ** 8 - 1) <= 0.01


def test_noise_other_semantically_float():
    torch.manual_seed(1337)
    for strength in np.arange(0, 0.55, 0.05):
        N, C, H, W = (2, 3, 3000, 3000)
        im = (torch.zeros(N, C, H, W) + 0.5)
        im = im.cuda()
        expected_std = math.sqrt((2 * strength) ** 2 / 12)
        expected_mean = 0.5

        a = {'noise': strength}
        augs = [a] * N
        noisy = cudaaugs.add_noise_other(im, augs)
        mean = noisy.mean()
        std = noisy.transpose(0, 1).contiguous().view(C, -1).std(dim=1, unbiased=False)
        assert abs(std[0] - expected_std) <= 0.001
        assert abs(mean - expected_mean) <= 0.0001


def test_noise_other_semantically_byte():
    for strength in np.arange(0, 0.55, 0.05):
        N, C, H, W = (2, 3, 3000, 3000)
        im = (torch.zeros(N, C, H, W)).byte() + 2 ** 7
        im = im.cuda()
        expected_std = math.sqrt((2 * strength * (2 ** 8 - 1)) ** 2 / 12)
        expected_mean = 2 ** 7
        a = {'noise': strength}
        augs = [a] * N
        noisy = cudaaugs.add_noise_other(im, augs)
        mean = noisy.sum().float() / (N * C * H * W)
        std = ((noisy.float() - mean).pow(2).transpose(0, 1).contiguous().view(C, -1).sum(dim=1) / (N * H * W)).sqrt()
        # std = noisy.transpose(0,1).reshape(C,-1).std(dim=1, unbiased = False)
        assert abs(std[0] - expected_std) / (2 ** 8 - 1) <= 0.01
        assert abs(mean - expected_mean) / (2 ** 8 - 1) <= 0.01


def _make_contrast_lut(a, b, n):
    lut = 1 / (1 + np.exp(np.linspace(a, b, n)))
    lut -= lut.min()
    lut /= lut.max()
    return lut


def test_gamma_byte():
    for gamma_color in np.arange(0.6, 1.45, 0.3):
        for gamma_gray in np.arange(0.8, 1.25, 0.05):
            for contrast in np.arange(-1.0, 1.0, 0.05):
                N, C, H, W = (1, 3, 300, 300)
                im = (torch.empty(N, C, H, W)).byte().random_()

                a = {'color': True, 'gamma_gray': gamma_gray, 'gamma_color': [gamma_color] * 3, 'contrast': contrast}
                augs = [a] * N
                gamma1 = cudaaugs.add_gamma(im.cuda(), augs)
                gamma2 = torch.tensor(
                    cpuaugs.add_gamma(im.squeeze().transpose(0, 2).numpy(), True, a['gamma_gray'], a['gamma_color'],
                                      a['contrast'])).transpose(0, 2).cuda()
                diff = gamma1.float() - (gamma2.float())

                # err = diff.abs().sum()/(C*H*W*(2**8-1))
                err = diff.abs().max() / (2 ** 8 - 1)
                # print(err)
                assert err.item() < 0.004


def test_nogamma_float():
    im = torch.empty(500, 3, 100, 100).uniform_().cuda()
    a = {'color': True, 'gamma_gray': 1, 'gamma_color': [1] * 3, 'contrast': 0}
    result = cudaaugs.add_gamma(im, [a] * 500)
    diff = im - result;
    err = diff.abs().mean()
    assert err.item() < 0.0001


def test_blur():
    for sigma in np.arange(0.00005, 0.2, 0.008):
        print("sigma: " + str(sigma))
        N, C, H, W = (1, 3, 500, 500)
        im = torch.rand(N, C, H, W)
        augs = [{'blur': sigma}]
        im_numpy = im.squeeze().transpose(0, 2).clone().numpy()
        blur1 = cudaaugs.add_blur(im, augs).squeeze()
        print(im_numpy.shape)
        print(im.shape)
        blur2 = torch.tensor(cpuaugs.add_blur(im_numpy, sigma)).transpose(0, 2)

        diff = blur1 - blur2
        err = diff.abs().mean()
        print(err)
        assert err < 0.01



def test_gamma_short():
    print("test")
    # torch.set_printoptions(profile="full")
    for gamma_color in np.arange(0.6, 1.45, 0.05):
        for gamma_gray in np.arange(0.8, 1.25, 0.05):
            for contrast in np.arange(0, 1.01, 0.1):
                N, C, H, W = (1, 3, 1, 2 ** 15 - 1)
                im = torch.tensor([[i for i in range(2 ** 15)]] * 3, dtype=torch.int16).view(1, 3, 1, -1)

                a = {'color': True, 'gamma_gray': gamma_gray, 'gamma_color': [gamma_color] * 3, 'contrast': contrast}
                augs = [a] * N

                gamma2 = torch.tensor(
                    cpuaugs.add_gamma(im.view(3, 1, -1).transpose(0, 2).numpy().astype('int16'), True, a['gamma_gray'],
                                      a['gamma_color'], a['contrast']).astype('float'))
                gamma2 = gamma2.transpose(0, 2).cuda().float().view(1, 3, 1, -1)
                gamma1 = cudaaugs.add_gamma(im.cuda(), augs, in_place=False).float()
                # gamma2 = torch.tensor(cpuaugs.add_gamma(im.clone().view(3,1,-1).transpose(0,2).contiguous().numpy().astype('uint16'), True, a['gamma_gray'], a['gamma_color'], a['contrast']).astype('float')).transpose(0,2).contiguous().cuda()
                diff = gamma1 - gamma2

                err = diff.abs().max() / (2 ** 15 - 1)

                assert err.item() < 4e-5  # maximum 1 bit difference is allowed

# def test_noise_rgb_semantically_byte_cpu():
#     torch.manual_seed(1337)
#     for strength in np.arange(0.05, 0.40, 0.05):
#         H, W, C = (3000, 3000, 3)
#         im = (torch.zeros(H, W, C)+128).byte().numpy()
#         expected_std = math.sqrt((2*strength*128)**2/12)
#         expected_mean = 127.5
#         noisy = torch.from_numpy(cpuaugs.add_noise_rgb(im, strength)).float()
#         mean = noisy[0].mean()
#         print(mean, expected_mean)
#         std = (noisy.transpose(0,2).contiguous().view(C,-1)[0]).std(unbiased = False)
#         print(std, expected_std)
#         print(abs(std-expected_std))
#         assert abs(std-expected_std)<= 0.001
#         assert abs(mean-expected_mean) <= 0.0005
#
# test_noise_rgb_semantically_byte_cpu()
