import os
import os.path as pt
import sys
from collections import defaultdict
from crumpets.broker import BufferWorker
# from crumpets import workers
from torch.nn import Module
import random

import msgpack
import msgpack_numpy

ROOT = pt.dirname(__file__)
parent = pt.abspath(pt.join(ROOT, os.pardir))

for p in (os.pardir, parent):
    try:
        sys.path.remove(p)
    except ValueError:
        pass
# print(sys.path)


import numpy as np
import torch
from torch.nn import Sequential
from crumpets.torch.trainer import Trainer
from torch.optim import SGD
from crumpets.torch.loss import *
from crumpets.torch.metrics import AccuracyMetric
from itertools import cycle
from crumpets.torch.policy import *
from crumpets.torch.dataloader import TorchTurboDataLoader


class Model(Module):
    def __init__(self):
        Module.__init__(self)
        self.l1 = torch.nn.Linear(2, 4, bias=True)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(4, 1, bias=True)

    def forward(self, sample):
        x = sample['input']
        x = self.l1(x)
        x = self.l2(x)
        # print('\n'+str(x) + '\n')
        x = self.l3(x)
        if 'label' in sample:
            sample['output'] = x.view_as(sample['label'])
        else:
            sample['output'] = x.squeeze()
        # print('\n'+str(sample) + '\n')
        # print(list(model.parameters()))
        # print("\n\n\n" + str(sample) + "\n\n\n")
        return sample


# class PolyPolicy(_LRPolicy):
# class SigmoidPolicy(_LRPolicy):
# class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
# class RampPolicy(_LRPolicy):

def test_PolyPolicy():
    model = Model()
    for exp in np.arange(0.1, 2, 0.15):
        for num_epochs in [100, 1000, 5000]:
            for initlr in np.arange(0.0001, 2, 0.2):
                optimizer = SGD(model.parameters(), lr=initlr, momentum=0.0)
                policy = PolyPolicy(optimizer, num_epochs=num_epochs, power=exp);
                for i in range(num_epochs):
                    policy.step()
                    assert abs(optimizer.param_groups[0]["lr"] - initlr * (1 - (i / num_epochs)) ** exp) <= 0.0000001
        print(num_epochs, exp)


def test_SigmoidPolicy():
    model = Model()
    for q in range(1, 10):
        for num_epochs in [100, 1000, 5000]:
            for initlr in np.arange(0.0001, 2, 0.2):
                optimizer = SGD(model.parameters(), lr=initlr, momentum=0.0)
                policy = SigmoidPolicy(optimizer, num_epochs=num_epochs, q=q);
                for i in range(num_epochs):
                    policy.step()

                    policy_lr = optimizer.param_groups[0]["lr"]
                    ground_lr = initlr / (1 + math.exp(q * (2 * i / num_epochs - 1)))
                    # print(policy_lr, ground_lr, abs(policy_lr-ground_lr))
                    # assert abs(policy_lr - ground_lr) <= 0.0000001
        print(num_epochs, q)


def test_ReduceLROnPlateau_constant():
    model = Model()
    optimizer = SGD(model.parameters(), lr=1.0, momentum=0.0)
    policy = ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=False);
    # for error in np.arange(100, 50, -0.1):
    for i, error in enumerate(np.arange(100, 1, -1)):
        policy.step(metrics=error, epoch=i)
        assert optimizer.param_groups[0]['lr'] == 1.0


def test_ReduceLROnPlateau_reducing():
    model = Model()
    optimizer = SGD(model.parameters(), lr=1.0, momentum=0.0)
    policy = ReduceLROnPlateau(optimizer, mode='min', patience=0, verbose=False, factor=0.99);
    old_lr = 999;
    # for error in np.arange(100, 50, -0.1):
    for i, error in enumerate(np.arange(1, 100, 1)):
        policy.step(metrics=error, epoch=i)
        assert optimizer.param_groups[0]['lr'] < old_lr
        old_lr = optimizer.param_groups[0]['lr']


def test_RampPolicy():
    model = Model()
    for num_epochs in [100, 1000, 5000]:
        for initlr in np.arange(0.0001, 2, 0.2):
            last_lr = -1
            optimizer = SGD(model.parameters(), lr=initlr, momentum=0.0)
            policy = RampPolicy(optimizer, ramp_epochs=num_epochs / 2);
            for i in range(num_epochs):
                policy.step()
                if i < num_epochs / 2 + 1:
                    assert last_lr < optimizer.param_groups[0]['lr']
                    last_lr = optimizer.param_groups[0]['lr']
                else:
                    assert optimizer.param_groups[0]['lr'] == last_lr
