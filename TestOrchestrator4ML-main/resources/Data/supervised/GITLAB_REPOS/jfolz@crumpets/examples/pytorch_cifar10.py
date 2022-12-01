"""
Example usage of crumpets to train a custom model on Cifar10. Less complex compared to resnet example, since
less parameters are considered (some are just set to their default value to make the example more intuitive).
Cifar10 can either be processed to be in msgpack format or directly downloaded, using Datadings.
This example is capable of using multiple gpus.
If no datadir is given a default sample of 10 images is used
while the loader is told that there are 2000 images to mimic a real dataset.
"""
from __future__ import print_function, unicode_literals, division

import os.path as pt
import sys

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datadings.reader import Cycler
from datadings.reader import MsgpackReader
from six import text_type
from torch.backends import cudnn
from torch.optim import SGD

from crumpets.workers import ClassificationWorker
from crumpets.presets import AUGMENTATION_TRAIN
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.torch.loss import CrossEntropyLoss
from crumpets.torch.metrics import AccuracyMetric
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.trainer import Trainer

ROOT = pt.dirname(__file__)
sys.path.insert(0, pt.join(ROOT, '..'))
DEFAULT_SAMPLE = pt.join(ROOT, '..', 'data', 'cifar10_sample')
CIFAR10_MEAN = (0.491 * 255, 0.482 * 255, 0.447 * 255)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, sample):
        x = sample['image']
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        sample['output'] = x
        return sample


def make_loader(
        file,
        batch_size,
        num_mini_batches,
        nworkers,
        image_rng=None,
        image_params=None,
        use_cuda=True,
        gpu_augmentation=True,
):
    reader = MsgpackReader(file)
    nsamples = len(reader) if pt.dirname(file) != DEFAULT_SAMPLE else 2000
    cycler = Cycler(reader)
    worker = ClassificationWorker(
        ((3, 32, 32), np.uint8, CIFAR10_MEAN),
        ((1,), np.int),
        image_params=image_params,
        image_rng=image_rng,
    )
    return TorchTurboDataLoader(
        cycler.rawiter(), batch_size,
        worker, nworkers,
        gpu_augmentation=gpu_augmentation,
        length=nsamples,
        num_mini_batches=num_mini_batches,
        device='cuda:0' if use_cuda else 'cpu:0',
    )


def make_policy(epochs, network, lr, momentum):
    optimizer = SGD([
        {'params': network.parameters(), 'lr': lr},
    ], momentum=momentum, weight_decay=1e-4)
    scheduler = PolyPolicy(optimizer, epochs, 1)
    return optimizer, scheduler


def main(
        datadir,
        outdir,
        batch_size,
        epochs,
        lr,
        cuda=True
):
    cudnn.benchmark = True
    if cuda:
        network = Net().cuda()
    else:
        network = Net()

    train = make_loader(
        pt.join(datadir, 'train.msgpack') if datadir else None,
        batch_size, 1, 4, use_cuda=cuda,
        gpu_augmentation=cuda, image_rng=AUGMENTATION_TRAIN
    )
    val = make_loader(
        pt.join(datadir, 'val.msgpack') if datadir else None,
        batch_size, 1, 4, use_cuda=cuda,
        gpu_augmentation=cuda
    )

    optimizer, policy = make_policy(epochs, network, lr, 0.9)

    loss = CrossEntropyLoss(target_key='label').cuda() if cuda else CrossEntropyLoss(target_key='label')
    trainer = Trainer(
        network, optimizer, loss, AccuracyMetric(),
        policy, None, train, val, outdir
    )
    with train:
        with val:
            trainer.train(epochs, 0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-b', '--batch-size',
        default=2,
        type=int,
        help='number of images in batch',
    )
    parser.add_argument(
        '-e', '--epochs',
        default=20,
        type=int,
        help='number of epochs to train',
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        default=0.0001,
        type=float,
        help='initial learning rate',
    )
    parser.add_argument(
        '--datadir',
        type=text_type,
        help='directory containing training and validation data in form of train.msgpack and val.msgpack',
    )
    parser.add_argument(
        '--outdir',
        default='.',
        type=text_type,
        help='output directory for snapshots and logs',
    )
    parser.add_argument(
        '--cpu',
        action="store_false",
        dest="cuda",
        help='activate to use cpu only for augmentations, forwarding and backwardings',
    )
    parser.set_defaults(
        cuda=True,
    )
    args, unknown = parser.parse_known_args()

    if args.datadir is None:
        args.datadir = DEFAULT_SAMPLE

    try:
        main(
            args.datadir,
            args.outdir,
            args.batch_size,
            args.epochs,
            args.lr,
            args.cuda,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
