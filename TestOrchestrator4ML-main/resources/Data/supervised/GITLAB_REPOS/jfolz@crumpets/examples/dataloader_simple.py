"""
Simple dataloader example without using datadings.
The example prepares the included tinyset dataset,
which is given in ./tinyset and in the form of imangenet's folder structure.
Afterwards crumpets TurboDataLoader is created and run through.
"""
import os.path as pt
import os
import sys
import io
import cv2

import numpy as np
from itertools import cycle

import msgpack
import msgpack_numpy

from crumpets.workers import ClassificationWorker
from crumpets.dataloader import TurboDataLoader


ROOT = pt.dirname(__file__)
sys.path.insert(0, pt.join(ROOT, '..'))


def prepare_dataset(dsdir):
    """
    We have to prepare our example dataset tinyset s.t. we have encoded images and labels.
    Crumpets default worker expect msgpack packed dictionaries.
    Thus we have to create an iterable of packed elements, which unpacked are of form: {'image': ..., 'label': ...}.

    :param dsdir: path to dataset directory
    :return: iterable of msgpack packed directories
    """
    iterable = []
    for cls_id, (cls_dir, _, imgs) in enumerate(list(os.walk(dsdir))[1:]):
        for img_path in imgs:
            with io.FileIO(pt.join(cls_dir, img_path)) as f:
                img = f.read()
            dic = {'image': img, 'label': cls_id}
            iterable.append(msgpack.packb(dic, use_bin_type=True, default=msgpack_numpy.encode))
    return iterable


def main(show=True, wait_key=2000):
    # parameters
    dataset_dir = pt.join(ROOT, 'tinyset')
    batch_size = 2
    epochs = 3
    nworkers = 2
    sample_size = (3, 224, 224)  # shape which processed images shall have, once outputed by the DataLoader

    # prepare iterable
    iterable = prepare_dataset(dataset_dir)
    num_samples = len(iterable)
    cycler = cycle(iterable)

    # create loader
    loader = TurboDataLoader(
        cycler, batch_size,
        ClassificationWorker(  # Other workers are available, such as SaliencyWorkers
            (sample_size, np.uint8),
            ((1, ), np.int),
        ),
        nworkers,
        length=num_samples,
    )

    # run
    with loader:  # It is important to use with, as that maintains the workers and such
        for epoch in range(epochs):
            print('Epoch {}'.format(epoch))
            for iteration, mini_batch in loader:
                print('Iteration {}'.format(iteration))
                for sample in mini_batch:
                    augmentation_info = sample['augmentation']  # can be used to track augmentation used per image
                    image = sample['image']
                    label = sample['label']
                    if show:
                        img = np.concatenate([*image], axis=2).transpose(1, 2, 0)
                        print("Label IDs found are {}".format(label.tolist()))
                        cv2.namedWindow('Sample', cv2.WINDOW_KEEPRATIO)
                        cv2.imshow('Sample', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(wait_key)
                    else:
                        print("{} imgs were found with labels {}"
                              .format(iteration, image.shape[0], label.cpu().numpy()))


if __name__ == '__main__':
    main()
