#!/usr/bin/env python

import click
import matplotlib.pyplot as plt
import numpy as np
import theanets

from utils import load_cifar, plot_layers, plot_images

K = 655  # this value of K retains 99% of the variance in the cifar images.

def pca(dataset):
    mean = dataset[:3000].mean(axis=0)

    theanets.log('computing whitening transform')
    x = dataset[:3000] - mean
    vals, vecs = np.linalg.eigh(np.dot(x.T, x) / len(x))
    vals = vals[::-1]
    vecs = vecs[:, ::-1]

    vals = np.sqrt(vals[:K])
    vecs = vecs[:, :K]

    def whiten(x):
        return np.dot(x, np.dot(vecs, np.diag(1. / vals)))

    def color(z):
        return np.dot(z, np.dot(np.diag(vals), vecs.T))

    return whiten, color


@click.command()
@click.option('--features', default=None, type=int, metavar='N',
              help='Train a model with NxN hidden features.')
def main(features):
    train, valid, _ = load_cifar()

    whiten, color = pca(train[0])

    feat = features or int(np.sqrt(2 * K))
    n = theanets.Autoencoder([K, feat ** 2, K])
    n.train(whiten(train), whiten(valid), input_noise=1, train_batches=313)

    plot_layers([
        color(n.find('hid1', 'w').get_value().T).T,
        color(n.find('out', 'w').get_value())], channels=3)
    plt.tight_layout()
    plt.show()

    valid = whiten(valid[:100])
    plot_images(color(valid), 121, 'Sample data', channels=3)
    plot_images(color(n.predict(valid)), 122,
                'Reconstructed data', channels=3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
