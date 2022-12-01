import os
import os.path as pt
import sys
from collections import defaultdict

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
import crumpets.torch.loss


# class L1Loss(nn.L1Loss):
def test_L1Loss():
    for _ in range(64):
        target = torch.rand(64).round()
        output = torch.rand(64)
        sample = {'output': output, 'target_image': target}
        loss = crumpets.torch.loss.L1Loss()
        losstorch = torch.nn.L1Loss(reduction="mean")
        lossvalue1 = losstorch(output, target)
        lossvalue2 = loss(sample)
        print(lossvalue1)
        assert lossvalue1 == lossvalue2


# class MSELoss(nn.MSELoss):
def test_MSELoss():
    for _ in range(64):
        target = torch.rand(64).round()
        output = torch.rand(64)
        sample = {'output': output, 'target_image': target}
        loss = crumpets.torch.loss.MSELoss()
        losstorch = torch.nn.MSELoss(reduction="mean")
        lossvalue1 = losstorch(output, target)
        lossvalue2 = loss(sample)
        print(lossvalue1)
        assert lossvalue1 == lossvalue2


# class NLLLoss(nn.NLLLoss):
def test_NLLLoss():
    for _ in range(64):
        target = torch.randint(0, 10, (64,))
        output = torch.rand(64, 10)
        sample = {'output': output, 'label': target}
        loss = crumpets.torch.loss.NLLLoss()
        losstorch = torch.nn.NLLLoss(reduction="mean")
        lossvalue1 = losstorch(output, target)
        lossvalue2 = loss(sample)
        print(lossvalue1)
        assert lossvalue1 == lossvalue2


# class CrossEntropyLoss(nn.CrossEntropyLoss):
def test_CrossEntropyLoss():
    for _ in range(64):
        target = torch.randint(0, 10, (64,))
        output = torch.rand(64, 10)
        sample = {'output': output, 'label': target}
        loss = crumpets.torch.loss.CrossEntropyLoss()
        losstorch = torch.nn.CrossEntropyLoss(reduction="mean")
        lossvalue1 = losstorch(output, target)
        lossvalue2 = loss(sample)
        print(lossvalue1)
        assert lossvalue1 == lossvalue2

# TODO class NSSLoss(nn.Module):
# def test_NSSLoss():
#     for _ in range(64):
#         target = torch.randint(0, 10,(64,))
#         output = torch.rand(64, 10)
#         sample = {'output': output, 'target_image': target}
#         loss = crumpets.torch.loss.NSSLoss()
#         losstorch = torch.nn.NSSLoss(reduction="mean")
#         lossvalue1 = losstorch(output, target)
#         lossvalue2 = loss(sample)
#         print(lossvalue1)
#         assert lossvalue1 == lossvalue2
