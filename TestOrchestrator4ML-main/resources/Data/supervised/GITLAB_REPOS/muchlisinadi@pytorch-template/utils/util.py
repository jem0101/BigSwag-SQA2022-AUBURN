import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
import torchvision

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

class ImageSaver:
    def __init__(self, key, writer=None):
        self.key = key
        self.writer = writer

    def make_grid(self, images):
        if torch.cuda.is_available():
            images = images.cpu()
        return torchvision.utils.make_grid(images, normalize=True)

    def update(self, images, step):
        images = self.make_grid(images)
        if self.writer is not None:
            self.writer.add_image(self.key, images, step)

def add_graph(model, input, writer=None):
    if model.__class__.__name__ == 'DataParallel':
        tag = list(model.children())[0].__class__.__name__
    else:
        tag = model.__class__.__name__
    writer.add_graph(tag, model, input, verbose=False)




