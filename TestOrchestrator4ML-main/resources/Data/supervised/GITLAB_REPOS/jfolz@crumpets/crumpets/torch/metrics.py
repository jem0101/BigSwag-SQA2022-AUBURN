from __future__ import print_function, division, absolute_import, unicode_literals

from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict

from six import with_metaclass

import numpy as np
from torch.nn import functional as F


class AverageValue(object):
    def __init__(self):
        self.total = 0
        self.steps = 0

    def value(self):
        return self.total / self.steps

    def __call__(self, value):
        self.steps += 1
        self.total += value
        return self.value()


class Metric(with_metaclass(ABCMeta)):
    """
    Abstract class which is to be inherited by every metric.
    As usual, this class is designed to handle crumpets dictionaries.

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the imput dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """
    def __init__(self, output_key='output', target_key='target_image', metric_key='metric'):
        self.metric_key = metric_key
        self.output_key = output_key
        self.target_key = target_key
        self.total = 0
        self.steps = 0

    def reset(self):
        self.total = 0
        self.steps = 0

    @abstractmethod
    def value(self):
        """
        implement to return the currently stored metric.
        :return: current metric
        """
        pass

    @abstractmethod
    def __call__(self, sample):
        """
        implement a call that processes given sample dictionary to compute a metric.
        :param sample: crumpets dictionary
        :return: metric
        """
        pass


class NoopMetric(Metric):
    """
    Provides the same API as a real metric but does nothing.
    Can be used where some metric-like object is required,
    but no actual metrics should be calculated.
    """
    def value(self):
        pass

    def __call__(self, sample):
        return {}


class ConfusionMatrix(Metric):
    """
    Computes the confusion matrix for given classification scores,
    i.e. predicted class probabilities.

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the input dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """
    def __init__(self, nclasses=10,
                 output_key='output', target_key='target_image', metric_key='confusion_matrix'):
        Metric.__init__(self, output_key, target_key, metric_key)
        self.nclasses = int(nclasses)
        self.reset()

    def reset(self):
        self.cmat = np.zeros((self.nclasses, self.nclasses), dtype=int)
        self.targets_per_class = np.zeros(self.nclasses)

    def value(self):
        return {self.metric_key: self.cmat}

    def __call__(self, sample):
        target = sample[self.target_key].data.cpu().numpy().flatten()
        output = sample[self.output_key].data.cpu().numpy()
        # old implementation in the comment below
        # self.targets_per_class[target] += 1
        # self.cmat[target, output.argmax(axis=-1)] += 1
        for i, j in zip(target, output.argmax(axis=-1)): #loop instead
            self.targets_per_class[i] +=1
            self.cmat[i,j] += 1
        return self.value()

    def get_true_false_positives(self):
        """
        Calculate the true positive and false positive rates per class
        :return: 2d-array. Cx3 array where the first column corresponds
                 to the true positives per class, the second column,
                 to the false positives per class and the last one,
                 the number of samples per class in total that have been
                 seen.
        """
        tps = self.cmat[np.eye(self.nclasses) == 1]
        fps = self.cmat.sum(axis=-1) - tps
        return np.c_[tps, fps, self.targets_per_class]


class AverageMetric(Metric):
    """
    Computes a simple average metric for given values inside the output.

    :param output_key: the key with which the output is found in the input dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """
    def __init__(self, output_key = "output", metric_key = "average_metric"):
        Metric.__init__(self, output_key=output_key, metric_key=metric_key)

    def value(self):
        return {self.metric_key: self.total / self.steps}

    def __call__(self, value):
        self.steps += 1
        self.total += value[self.output_key]
        return self.value()


class AccuracyMetric(Metric):
    """
    Computes the top-k accuracy metric for given classification scores,
    i.e. predicted class probabilities.
    The metric is computed as {1 if target_i in top_k_predicted_classes_i else 0 for all i in n} / n

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the input dictionary
    """
    def __init__(self, top_k=1,
                 output_key='output', target_key='label'):
        Metric.__init__(self, output_key, target_key, None)
        try:
            top_k[0]
        except (AttributeError, TypeError):
            top_k = (top_k,)
        self.top_k = top_k
        self.output_key = output_key
        self.target_key = target_key
        self.reset()

    def reset(self):
        self.correct = defaultdict(lambda: 0)
        self.n = 0

    def value(self):
        return {k: v.item() / self.n for k, v in self.correct.items()}

    def __call__(self, sample):
        n = sample[self.output_key].size()[0]
        self.n += n
        target = sample[self.target_key].data.view(n, 1)
        output = sample[self.output_key].data
        predictions = output.sort(1, descending=True)[1]
        for k in self.top_k:
            self.correct['top-%d acc' % k] += \
                predictions[:, :k].eq(target).sum()
        return self.value()


class MSELossMetric(Metric):
    """
    Computes the mean squared error

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the input dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """
    def __init__(self, output_key = "output", target_key = "target_image", metric_key = "mse"):
        Metric.__init__(self, output_key=output_key, target_key=target_key, metric_key = metric_key)

    def value(self):
        return {self.metric_key: self.total / self.steps}

    def __call__(self, sample):
        self.steps += 1
        self.total += F.mse_loss(
            sample[self.output_key],
            sample[self.target_key],
            True
        ).item()
        return self.value()


class NSSMetric(Metric):
    """
    Computes the Normalized Scanpath Saliency (NSS) by Bylinskii et. al. (https://arxiv.org/pdf/1604.03605.pdf)

    :param output_key: the key with which the output is found in the input dictionary
    :param target_key: the key with which the target is found in the input dictionary
    :param metric_key: the key with which the metric is to be stored in the output dictionary
    """
    def __init__(self, output_key = "output", target_key = "target_image", metric_key = "nss"):
        Metric.__init__(self, output_key=output_key, target_key=target_key, metric_key = metric_key)
    def value(self):
        return {self.metric_key: self.total / self.steps}

    def __call__(self, sample):
        output = sample[self.output_key].data
        target = sample[self.target_key].data
        n, c, h, w = output.size()
        vectorized = output.view(n, c, h*w)
        mean = vectorized.mean(2).view(n, c, 1, 1)
        std = vectorized.std(2).view(n, c, 1, 1)
        nss = output - mean
        nss /= std + 1e-5
        nss = nss.mul(target).sum() / target.sum()
        self.total += nss / n
        self.steps += 1
        return self.value()


class CombinedMetric(object):
    """
    A simple meta metric. Given metric instances, returns a collection of them.

    :param children: list of metric class instances
    """
    def __init__(self, children):
        self.children = children

    def __call__(self, sample):
        metrics = {}
        for child in self.children:
            metrics.update(child(sample))
        return metrics
