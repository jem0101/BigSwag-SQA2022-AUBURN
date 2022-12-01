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


import numpy as np
import torch
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.dataloader import TurboDataLoader
from crumpets.workers import ImageWorker
from datadings.reader import MsgpackReader as CAT2000Reader
from datadings.reader import Cycler
from crumpets.presets import IMAGENET_MEAN as MEAN
from crumpets.rng import MixtureRNG
from crumpets.presets import AUGMENTATION_ALL


# TODO write version that doesn't use datadings. See examples.dataloader_datadings
def test_TorchTurbodataloader_GPU_augs():
    for i in range(5):
        reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
        nsamples = len(reader)
        cycler = Cycler(reader)
        rng = MixtureRNG(
            prob=1,
            color_range=(-0.25, 0.25),
            blur_range=(0.2 / 448, 2.5 / 448),
        )
        worker = ImageWorker(
            ((3, 256, 256), np.uint8, MEAN),
            dict(scale_mode='shortest'),
            image_rng=rng,
            gpu_augmentation=True,
        )
        loader = TorchTurboDataLoader(
            cycler.rawiter(), 4, worker, 1,
            length=nsamples,
            num_mini_batches=1,
            device='cuda:0',
            gpu_augmentation=True
        )
        # print(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
        count = 0
        with loader:
            for iteration, mini_batch in loader:
                for sample in mini_batch:
                    count += sample['image'].shape[0]
                    assert sample['image'].shape == (4, 3, 256, 256)
        assert count == 12
        # loader.stop()


def test_TorchTurbodataloader_CPU_augs():
    for i in range(5):
        reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
        nsamples = len(reader)
        cycler = Cycler(reader)
        rng = MixtureRNG(
            prob=1,
            color_range=(-0.25, 0.25),
            blur_range=(0.2 / 448, 2.5 / 448),
        )
        worker = ImageWorker(
            ((3, 256, 256), np.uint8, MEAN),
            dict(scale_mode='shortest'),
            image_rng=rng,
            gpu_augmentation=False,
        )
        loader = TorchTurboDataLoader(
            cycler.rawiter(), 4, worker, 1,
            length=nsamples,
            num_mini_batches=1,
            device='cpu',
            gpu_augmentation=False
        )
        # print(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
        count = 0
        with loader:
            for iteration, mini_batch in loader:
                for sample in mini_batch:
                    count += sample['image'].shape[0]
                    assert sample['image'].shape == (4, 3, 256, 256)
        assert count == 12
        # loader.stop()


def test_Turbodataloader():
    reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    nsamples = len(reader)
    cycler = Cycler(reader)
    rng = MixtureRNG(
        prob=1,
        blur_range=(0.2 / 448, 2.5 / 448),
    )
    worker = ImageWorker(
        ((3, 256, 256), np.uint8, MEAN),
        dict(scale_mode='shortest'),
        image_rng=rng,
        gpu_augmentation=True,
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
                # print(sample)
                assert sample['image'].shape == (4, 3, 256, 256)
    loader.stop()


def test_compare_dataloaders():
    reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    reader2 = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    nsamples = len(reader)
    cycler = Cycler(reader)
    cycler2 = Cycler(reader2)
    worker = ImageWorker(
        ((3, 256, 256), np.uint8, MEAN),
        dict(scale_mode='shortest'),
        image_rng=AUGMENTATION_ALL,
        gpu_augmentation=True,
    )
    loader = TurboDataLoader(
        cycler.rawiter(), 4, worker, 1,
        length=nsamples,
        num_mini_batches=1,
    )
    worker2 = ImageWorker(
        ((3, 256, 256), np.uint8, MEAN),
        dict(scale_mode='shortest'),
        image_rng=AUGMENTATION_ALL,
        gpu_augmentation=True,
    )
    loader2 = TurboDataLoader(
        cycler2.rawiter(), 4, worker2, 1,
        length=nsamples,
        num_mini_batches=1,
    )
    with loader, loader2:
        for ((iteration, mini_batch), (iteration2, mini_batch2)) in zip(loader, loader2):
            for (sample, sample2) in zip(mini_batch, mini_batch2):
                im1 = torch.from_numpy(sample['image']).float()
                im2 = torch.from_numpy(sample2['image']).float()
                diff = im1 - im2
                abs = diff.abs().sum() / im1.view(-1).shape[0]
                # print(diff)
                # print(abs)
                assert sample['image'].shape == (4, 3, 256, 256)
                assert abs.item() > 20
    loader.stop()
    loader2.stop()


def test_compare_dataloader_no_augmentation():
    reader = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    reader2 = CAT2000Reader(pt.join(ROOT, '..', 'data', 'CAT2000_sample.msgpack'))
    nsamples = len(reader)
    cycler = Cycler(reader)
    cycler2 = Cycler(reader2)
    worker = ImageWorker(
        ((3, 256, 256), np.uint8, MEAN),
        dict(scale_mode='shortest'),
        gpu_augmentation=True,
    )
    loader = TurboDataLoader(
        cycler.rawiter(), 4, worker, 1,
        length=nsamples,
        num_mini_batches=1,
    )
    worker2 = ImageWorker(
        ((3, 256, 256), np.uint8, MEAN),
        dict(scale_mode='shortest'),
        gpu_augmentation=True,
    )
    loader2 = TurboDataLoader(
        cycler2.rawiter(), 4, worker2, 1,
        length=nsamples,
        num_mini_batches=1,
    )
    with loader, loader2:
        for ((iteration, mini_batch), (iteration2, mini_batch2)) in zip(loader, loader2):
            for (sample, sample2) in zip(mini_batch, mini_batch2):
                im1 = torch.from_numpy(sample['image']).float()
                im2 = torch.from_numpy(sample2['image']).float()
                diff = im1 - im2
                abs = diff.abs().sum() / im1.view(-1).shape[0]
                # print(diff)
                # print(abs)
                assert sample['image'].shape == (4, 3, 256, 256)
                assert abs.item() == 0
