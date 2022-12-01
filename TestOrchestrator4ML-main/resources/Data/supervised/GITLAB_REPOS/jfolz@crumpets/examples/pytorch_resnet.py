"""
Example usage of crumpets to train ResNet on ImageNet.
ImageNet can either be processed to be in msgpack format or directly downloaded, using Datadings.
This example is capable of using multiple gpus.
If no datadir is given a default sample of 10 images is used
while the loader is told that there are 2000 images to mimic a real dataset.
"""
from __future__ import print_function, unicode_literals, division

import os.path as pt
import sys

import numpy as np
from datadings.reader import Cycler
from datadings.reader import MsgpackReader
from six import text_type
from torch.backends import cudnn
from torch.optim import SGD
from torchvision.models import resnet

from crumpets.workers import ClassificationWorker
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.torch.loss import CrossEntropyLoss
from crumpets.torch.metrics import AccuracyMetric
from crumpets.presets import IMAGENET_MEAN
from crumpets.presets import AUGMENTATION_TRAIN
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.trainer import Trainer
from crumpets.torch.utils import resume
from crumpets.torch import is_cpu_only
from crumpets.torch.utils import Unpacker
from crumpets.torch.utils import Normalize


ROOT = pt.dirname(__file__)
sys.path.insert(0, pt.join(ROOT, '..'))
DEFAULT_SAMPLE = pt.join(ROOT, '..', 'data', 'imagenet_sample')


def make_loader(
        file,
        batch_size,
        num_mini_batches,
        nworkers,
        image_rng=None,
        image_params=None,
        device='cuda:0',
        gpu_augmentation=True,
):
    image_params = {'scale_mode': 'longest'}.update(image_params or {})
    reader = MsgpackReader(file)
    nsamples = len(reader) if pt.dirname(file) != DEFAULT_SAMPLE else 2000
    cycler = Cycler(reader)
    worker = ClassificationWorker(
        ((3, 224, 224), np.uint8, IMAGENET_MEAN),
        ((1,), np.int),
        image_params=image_params or {},
        image_rng=image_rng,
    )
    return TorchTurboDataLoader(
        cycler.rawiter(), batch_size,
        worker, nworkers,
        gpu_augmentation=gpu_augmentation,
        length=nsamples,
        num_mini_batches=num_mini_batches,
        device=device,
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
        num_mini_batches,
        nworkers,
        epochs,
        snapshot,
        finetune,
        lr,
        arch='resnet50',
        torch_device='cuda:0',
):
    cudnn.benchmark = True
    try:
        fun = getattr(resnet, arch)
    except AttributeError:
        raise ValueError('architecture "%s" does not exist' % arch)
    network = Unpacker(Normalize(fun())).to(torch_device)

    iteration = 0
    if finetune or snapshot:
        state = resume(finetune or snapshot, network, None)
        if snapshot:
            iteration = state['iteration']

    train = make_loader(
        pt.join(datadir, 'train.msgpack') if datadir else None,
        batch_size, num_mini_batches, nworkers, device=torch_device,
        gpu_augmentation=not is_cpu_only(torch_device),
        image_rng=AUGMENTATION_TRAIN,
    )
    val = make_loader(
        pt.join(datadir, 'val.msgpack') if datadir else None,
        batch_size, num_mini_batches, nworkers, device=torch_device,
        gpu_augmentation=not is_cpu_only(torch_device),
        image_params={'scale': 256/224},
    )

    optimizer, policy = make_policy(epochs, network, lr, 0.9)
    if snapshot:
        state = resume(snapshot, None, optimizer)
        train.iterations = state['iteration']
    start_epoch = iteration // train.epoch_iterations

    loss = CrossEntropyLoss(target_key='label')
    if not is_cpu_only(torch_device):
        loss = loss.cuda()
    trainer = Trainer(
        network, optimizer, loss, AccuracyMetric(),
        policy, None, train, val, outdir
    )
    with train:
        with val:
            trainer.train(epochs, start_epoch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-b', '--batch-size',
        default=2,  # TODO set to 128
        type=int,
        help='number of images in batch',
    )
    parser.add_argument(
        '-m', '--minibatches',
        default=1,
        type=int,
        help='number of mini batches per batch',
    )
    parser.add_argument(
        '-w', '--workers',
        default=1,
        type=int,
        help='number of workers',
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
        '-s', '--snapshot',
        type=text_type,
        help='resume training from this snapshot',
    )
    parser.add_argument(
        '-f', '--finetune',
        type=text_type,
        help='fine-tune from this snapshot',
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
        '--arch',
        default='resnet50',
        choices=resnet.model_urls.keys(),
        help='which architecture to train',
    )
    parser.add_argument(
        '--cpu',
        action="store_false",
        dest="cuda",
        help='activate to use cpu only for augmentations, forwarding and backwardings',
    )
    parser.add_argument(
        '--devices', '-d',
        default=['cuda:0'],
        type=text_type,
        nargs='+',
        help='list of cuda devices to use if cuda is activated, e.g. "cuda:0", '
             'counts only visible devices; None for all available',
    )
    parser.set_defaults(
        cuda=True,
    )
    args, unknown = parser.parse_known_args()

    if args.layer not in [18, 34, 50, 101]:
        parser.print_usage()
        print('Layer must be one of {18, 34, 50, 101}')
        sys.exit(1)

    if args.datadir is None:
        args.datadir = DEFAULT_SAMPLE

    try:
        main(
            args.datadir,
            args.outdir,
            args.batch_size,
            args.minibatches,
            args.workers,
            args.epochs,
            args.snapshot,
            args.finetune,
            args.lr,
            arch=args.arch,
            torch_device=args.devices if args.devices != ["None"] else None,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print()
