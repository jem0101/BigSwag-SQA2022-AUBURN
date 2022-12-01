from __future__ import print_function, division, absolute_import

from collections import OrderedDict

from six import text_type
from six import binary_type

import torch
from torch import nn
from ..presets import IMAGENET_MEAN
from ..presets import IMAGENET_STD


def save(path, iteration, model, optimizer, **kwargs):
    model_state = None
    optimizer_state = None
    if model is not None:
        model_state = model.state_dict()
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
    torch.save(
        dict(iteration=iteration,
             model_state=model_state,
             optimizer_state=optimizer_state,
             **kwargs),
        path
    )


def resume(path, model, optimizer):
    """
    Given parameters, extracts a training state, i.e. initializes a network and optimizer.

    :param path: path to a pytorch snapshot (including model and optimizer states)
    :param model: a network architecture for that the extracted weights are applied to
    :param optimizer: an optimizer for which the extracted optimizer parameters are applied to
    :return: the loaded snapshot
    """
    snapshot = torch.load(path)
    model_state = snapshot.pop('model_state', None)
    optimizer_state = snapshot.pop('optimizer_state', None)
    if model is not None and model_state is not None:
        model.load_state_dict(model_state)
    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return snapshot


def other_type(s):
    if isinstance(s, text_type):
        return s.encode('utf-8')
    elif isinstance(s, binary_type):
        return s.decode('utf-8')


def try_dicts(k, *ds):
    for d in ds:
        v = d.get(k)
        if v is not None:
            return v
    raise KeyError(k)


def try_types(k, *ds):
    try:
        return try_dicts(k, *ds)
    except KeyError:
        return try_dicts(other_type(k), *ds)


def filter_state(own_state, state_dict):
    return OrderedDict((k, try_types(k, state_dict, own_state))
                       for k in own_state)


class Normalize(nn.Module):
    def __init__(self, module, mean=IMAGENET_MEAN, std=IMAGENET_STD, grad=False):
        super(Normalize, self).__init__()
        self.module = module
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        invstd = [1.0 / v for v in std]
        self.register_buffer('invstd', torch.tensor(invstd).view(1, -1, 1, 1))
        self.grad = grad

    def forward(self, x):
        if self.grad:
            x = x.float()
            x = x.sub(self.mean).mul(self.invstd)
        else:
            with torch.no_grad():
                x = x.float()
                x.sub_(self.mean).mul_(self.invstd)
        return x


class Unpacker(nn.Module):
    def __init__(self, module, input_key='image', output_key='output'):
        super(Unpacker, self).__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.module = module

    def forward(self, sample, *args, **kwargs):
        x = sample[self.input_key]
        x.data = x.data.float()
        x = self.module.forward(x)
        sample[self.output_key] = x
        return sample