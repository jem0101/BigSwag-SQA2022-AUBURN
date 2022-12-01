#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import tempfile
import os
import gzip
import urllib
import cPickle

from . import Dataset
import numpy as np

logging = logging.getLogger(__name__)

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

class MnistDataset(Dataset):

    def __init__(self, for_autoencoder=False, more_data=False):
        """
        Load MNIST dataset.
        :param for_autoencoder: if this dataset is used for autoencoders
        """
        self.for_autoencoder = for_autoencoder
        self._target_size = 10
        logging.info("loading minst data")
        path = os.path.join(tempfile.gettempdir(), "mnist.pkl.gz")
        if not os.path.exists(path):
            logging.info("downloading minst data")
            urllib.urlretrieve (MNIST_URL, path)
        self._train_set, self._valid_set, self._test_set = cPickle.load(gzip.open(path, 'rb'))
        self._more_data = more_data
        logging.info("[mnist] training data size: %d" % len(self._train_set[0]))
        logging.info("[mnist] valid data size: %d" % len(self._valid_set[0]))
        logging.info("[mnist] test data size: %d" % len(self._test_set[0]))

    def train_set(self):
        data, target = self._train_set
        if self._more_data:
            data = np.vstack([data, data, data, data, data])
            target = np.concatenate([target, target, target, target, target])
        if self.for_autoencoder:
            return map(lambda d: [d], data)
        else:
            return zip(data,  np.array(target))

    def valid_set(self):
        data, target = self._valid_set
        if self.for_autoencoder:
            return map(lambda d: [d], data)
        else:
            return zip(data,  np.array(target))

    def test_set(self):
        data, target = self._test_set
        if self.for_autoencoder:
            return map(lambda d: [d], data)
        else:
            return zip(data,  np.array(target))