from __future__ import print_function, division

import math

from torch.optim import Optimizer
from torch.optim import lr_scheduler


class _LRPolicy(lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        lr_scheduler.LambdaLR.__init__(self, optimizer, 1, last_epoch)
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    # noinspection PyMethodOverriding
    def step(self, epoch=None, metrics=None):
        lr_scheduler.LambdaLR.step(self, epoch)


class NoopPolicy(object):
    """
    Just a noop Policy. Use it when you don't want to modify the lr

    """
    def step(self, *args, **kwargs):
        pass


class PolyPolicy(_LRPolicy):
    """
    A policy that can be described as a polynomial.

    :param optimizer: an optimizer object
    :param num_epochs: the number of epochs that this policy is defined for. Don't use it longer than that, because this might cause unexpected behaviour
    :param power: power value
    :param last_epoch: The current state of the policy. This can be used to set the initial state of the policy for instance to change the policy during training.
    """
    def __init__(self, optimizer, num_epochs=1, power=0.5, last_epoch=-1):
        self.power = power
        self.num_epochs = num_epochs
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        _LRPolicy.__init__(self, optimizer, last_epoch)

    def get_lr(self):
        i = self.last_epoch
        p = self.power
        n = self.num_epochs
        return [base_lr * (1 - i/n) ** p for base_lr in self.base_lrs]


class SigmoidPolicy(_LRPolicy):
    """
    A policy that can be described as a sigmoid. It can be described using the formula base_lr / (1 + math.exp(self.q * x), where x is last_epoch/num_epochs - 1

    :param optimizer: an optimizer object
    :param num_epochs: the number of epochs that this policy is defined for. Don't use it longer than that, because this might cause unexpected behaviour
    :param q: q value to describe the behaviour of the policy.
    :param last_epoch: The current state of the policy. This can be used to set the initial state of the policy for instance to change the policy during training.
    """
    def __init__(self, optimizer, num_epochs=1, q=6, last_epoch=-1):
        self.q = q
        self.num_epochs = num_epochs
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        _LRPolicy.__init__(self, optimizer, last_epoch)

    def get_lr(self):
        x = 2 * self.last_epoch / self.num_epochs - 1
        return [base_lr / (1 + math.exp(self.q * x))
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    """
    A policy that reduces the learning rate when the training progress reaches a plateau. It inherits from torch.optim.lr_scheduler.ReduceLROnPlateau and because of that shares the same interface
    """
    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        if epoch % 1 == 0:
            lr_scheduler.ReduceLROnPlateau.step(self, metrics,epoch)


class RampPolicy(_LRPolicy):
    """
    This Policy increases the learning rate step by step

    :param optimizer: an optimizer object
    :param ramp_epochs: the value where the plateau is reached
    :param last_epoch: The current state of the policy. This can be used to set the initial state of the policy for instance to change the policy during training.
    """
    def __init__(self, optimizer, ramp_epochs=1, last_epoch=-1):
        self.ramp_epochs = ramp_epochs
        _LRPolicy.__init__(self, optimizer, last_epoch)

    def get_lr(self):
        pos = self.last_epoch / self.ramp_epochs
        return [lr*pos for lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch > self.ramp_epochs:
            epoch = self.ramp_epochs
        _LRPolicy.step(self, epoch, metrics)


# class EstimatePolicy(_LRPolicy):
#     def __init__(self, optimizer,
#                  min_multiplier=1e-3, max_multiplier=1e3, stepsize=2, max_lr=0.1,
#                  last_epoch=-1):
#         _LRPolicy.__init__(self, optimizer, last_epoch)
#         self.base_lrs = [lr if lr else initial_lr
#                          for lr in self.base_lrs]
#         self.min_multiplier = min_multiplier
#         self.max_multiplier = max_multiplier
#         self.stepsize = stepsize
#         self.max_lr = max_lr
#         self.estimating = True
#         self.recorded_metrics = []
#         self.estimated_lrs = list(self.base_lrs)
#
#     def better(self):
#         try:
#             sl, l = self.recorded_metrics[-2:]
#         except IndexError:
#             return True
#         return l < sl
#
#     def get_lr(self):
#         return self.base_lrs
#
#     def step(self, metrics, epoch=None):
#         if epoch is None:
#             epoch = self.last_epoch = self.last_epoch + 1
#         if epoch % 1 == 0:
#             self.estimating = True
#             self.recorded_metrics = []
#             self.estimated_lrs = [lr * self.min_multiplier
#                                   for lr in self.base_lrs]
#         if self.estimating:
#             self.recorded_metrics.append(metrics)
#             _LRPolicy.step(self, metrics, epoch)

