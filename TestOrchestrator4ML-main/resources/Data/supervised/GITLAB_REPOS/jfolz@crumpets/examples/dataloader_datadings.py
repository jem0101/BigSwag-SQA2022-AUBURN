"""
A more detailed Dataloader example using datadings written msgpack files and custom augmentation.
Also activates GPU support and thus uses a Torch version of the DataLoader with torch tensors instead of np ndarrays.
For more explanation of how datadings works and how to create those msgpack files, have a look at
    https://git.opendfki.de/ML/core/datadings/
"""
import os.path as pt
import sys
import cv2

import numpy as np
import torch

from datadings.reader import Cycler
from datadings.reader import MsgpackReader

from crumpets.workers import ClassificationWorker
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.torch.utils import ParallelApply
from crumpets.presets import AUGMENTATION_TRAIN


ROOT = pt.dirname(__file__)
sys.path.insert(0, pt.join(ROOT, '..'))


class Identity(torch.nn.Module):
    """
    This simulates some module.
    """
    def forward(self, input):
        return input


def main(show=True, wait_key=2000):
    # parameters
    dataset_file = pt.join(ROOT, '..', 'data', 'ILSVRC2012_sample.msgpack')
    batch_size = 4
    epochs = 3
    nworkers = 2
    sample_size = (3, 224, 224)  # shape which processed images shall have, once outputed by the DataLoader

    # prepare iterable
    reader = MsgpackReader(dataset_file)
    num_samples = len(reader)
    cycler = Cycler(reader)

    # create loader
    loader = TorchTurboDataLoader(  # TorchTurboDataLoader outputs torch tensors instead of numpy arrays
        cycler.rawiter(), batch_size,
        ClassificationWorker(
            (sample_size, np.uint8), ((1, ), np.int),
            image_rng=AUGMENTATION_TRAIN,
        ),
        nworkers,
        gpu_augmentation=True,  # this enables gpu support for augmentations
        length=num_samples,
        num_mini_batches=2,
        device='cuda:0',  # this puts tensors on the gpu, note that this changes the output format from dict to list
    )

    # Create a model and wrap a ParallelApply module around it,
    # since DataScatter module is used in loader if multiple devices are available.
    # Note that this is incorrect if devices is set to a single value (not iterable!)
    model = ParallelApply(Identity())

    # run
    with loader:  # It is important to use with, as that maintains the workers and such
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            for iteration, mini_batch in loader:
                print('Iteration {}'.format(iteration))
                for n, sample in enumerate(mini_batch):
                    print('Minibatch {}'.format(n))
                    sample = model(sample)
                    augmentation_info = sample['augmentation']  # can be used to track augmentation used per image
                    image = sample['image']
                    label = sample['label']
                    if show:
                        img = np.concatenate([*image.cpu().numpy()], axis=2).transpose(1, 2, 0)
                        print("Label IDs found are {}".format(label.cpu().numpy().tolist()))
                        cv2.namedWindow('Sample', cv2.WINDOW_KEEPRATIO)
                        cv2.imshow('Sample', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(wait_key)
                    else:
                        print("{} imgs were found with labels {}"
                              .format(iteration, image.shape[0], label.cpu().numpy()))


if __name__ == '__main__':
    main()
