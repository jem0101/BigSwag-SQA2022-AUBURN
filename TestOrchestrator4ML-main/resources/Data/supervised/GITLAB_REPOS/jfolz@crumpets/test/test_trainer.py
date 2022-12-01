import os
import os.path as pt
import sys
from crumpets.broker import BufferWorker
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

import numpy as np
import torch
from crumpets.torch.trainer import Trainer
from torch.optim import SGD
from crumpets.torch.loss import *
from itertools import cycle
from crumpets.torch.dataloader import TorchTurboDataLoader


def prepare_dataset(data, labels):
    iterable = []
    for elem, label in zip(data, labels):
        dic = {'image': elem, 'label': label}
        iterable.append(msgpack.packb(dic, use_bin_type=True, default=msgpack_numpy.encode))
    return iterable


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


class TestWorker(BufferWorker):
    def __init__(self, input, label, **kwargs):
        BufferWorker.__init__(self, **kwargs)
        self.add_buffer("input", input)
        self.add_buffer("label", label)

    def prepare(self, sample, batch, buffers):
        buffers['input'][...] = sample['image']
        buffers['label'][...] = sample['label']


def create_dataloader(data, batch_size, worker, nworkers):
    return TorchTurboDataLoader(cycle(data), batch_size, worker, nworkers, num_mini_batches=1, length=len(data),
                                gpu_augmentation=False, device="cuda:0")


def test_trainer():
    torch.manual_seed(13370)
    torch.cuda.manual_seed(1598)
    random.seed(30)
    torch.cuda.manual_seed_all(1)
    np.random.seed(9999)
    model = Model().cuda()
    data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape(4, 1, 2)
    truth = np.array([0, 1, 1, 0]).reshape(4, 1)
    worker = TestWorker(((1, 2), np.float32, 0), ((1,), np.float32, 0))
    dl_image = create_dataloader(prepare_dataset(data, truth), 2, worker, 1)
    # dl_image = list(enumerate(data))
    # dl_ground = list(enumerate(truth))
    optimizer = SGD(model.parameters(), lr=0.2, momentum=0.0)
    # policy = PolyPolicy(optimizer, num_epochs=10, last_epoch=0)#SigmoidPolicy(optimizer, num_epochs=100, q=6)
    loss = MSELoss(output_key="output", target_key="label")  # L1Loss(output_key="output", target_key="label" )
    with dl_image:
        trainer = Trainer(model, optimizer, loss, None, None, None, dl_image, dl_image, None, snapshot_interval=999999)
        trainer.train(num_epochs=100, start_epoch=1)
    data_torch = torch.from_numpy(data).squeeze().cuda().float()
    result = model({'input': data_torch})['output'].round().cpu()
    assert (result == torch.tensor([0, 1, 1, 0], dtype=torch.float32)).sum().item() == 4
