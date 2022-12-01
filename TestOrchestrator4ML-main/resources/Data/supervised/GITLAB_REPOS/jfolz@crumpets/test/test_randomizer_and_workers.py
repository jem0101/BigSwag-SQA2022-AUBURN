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

from crumpets.workers.saliency import SaliencyWorker
import numpy as np
import torch
from crumpets.dataloader import TurboDataLoader
from datadings.reader import MsgpackReader as CAT2000Reader
from datadings.reader import Cycler
from crumpets.presets import IMAGENET_MEAN as MEAN
from crumpets.torch.randomizer import Randomizer
from torch import nn


class Identity(nn.Module):
    def forward(x):
        return x


def test_randomizer_3_channels():
    ran = Randomizer(net=Identity)
    data = {
        'image': torch.tensor(5 * 3 * [[i for i in range(256)]], dtype=torch.uint8).view(5, 3, 16, 16).cuda(),
        'augmentation': 5 * [
            {'gamma_gray': 1.2, 'gamma_color': [1.1, 1.3, 1.0], 'contrast': 0.1, 'noise': 0.2, 'blur': 0.05,
             "gpu_augmentation": True}]
    }

    newdata = ran.forward(data)
    diff = data['image'].float() - newdata['image'].float()
    # print (data['image'].float())
    assert diff.abs().sum().item() > 0.1


def test_randomizer_1_channel():
    ran = Randomizer(net=Identity)
    data = {
        'image': torch.tensor(5 * [[i for i in range(256)]], dtype=torch.uint8).view(5, 1, 16, 16).cuda(),
        'augmentation': 5 * [{'gamma_gray': 1.2, 'contrast': 0.1, 'noise': 0.2, 'blur': 0.05, "gpu_augmentation": True}]
    }

    newdata = ran.forward(data)
    diff = data['image'].float() - newdata['image'].float()
    # print (data['image'].float())
    assert diff.abs().sum().item() > 0.1


def test_randomizer_no_augmentation():
    ran = Randomizer(net=Identity)
    data = {
        'image': torch.tensor(5 * 3 * [[i for i in range(256)]], dtype=torch.uint8).view(5, 3, 16, 16).cuda(),
        'augmentation': 5 * [
            {'gamma_gray': 1.2, 'gamma_color': [1.1, 1.3, 1.0], 'contrast': 0.1, 'noise': 0.2, 'blur': 0.05,
             "gpu_augmentation": False}]
    }

    newdata = ran.forward(data)
    diff = data['image'].float() - newdata['image'].float()
    # print (data['image'].float())
    assert diff.abs().sum().item() == 0


def test_saliency_worker():
    reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    nsamples = len(reader)
    cycler = Cycler(reader)
    worker = SaliencyWorker(
        ((3, 256, 256), np.uint8, MEAN), ((3, 256, 256), np.float32),
        dict(scale_mode='shortest'),
        {"blur_range": (0.2 / 448, 2.5 / 448), 'prob': 1}
    )
    loader = TurboDataLoader(
        cycler.rawiter(), 4, worker, 1,
        length=nsamples,
        num_mini_batches=1,
    )
    # print(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    with loader:
        for iteration, mini_batch in loader:
            for sample in mini_batch:
                print(sample)
                assert sample['image'].shape == (4, 3, 256, 256)
    loader.stop()
