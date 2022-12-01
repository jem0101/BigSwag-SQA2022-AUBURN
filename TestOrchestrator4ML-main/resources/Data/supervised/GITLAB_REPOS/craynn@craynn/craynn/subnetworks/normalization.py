import inspect

from ..nonlinearities import linear
from ..layers import batch_norm

__all__ = [
  'batch_normed'
]

def _batch_normed(layer, *bn_args, **bn_kwargs):
  signature = inspect.signature(layer)
  def new_layer(*args, **kwargs):
    parameters = signature.bind(*args, **kwargs)
    parameters.apply_defaults()
    arguments = parameters.arguments

    activation = arguments.pop('activation', linear())

    new_model = layer(**arguments, activation=linear())

    def f(*incoming):
      net = new_model(*incoming)
      return batch_norm(*bn_args, **bn_kwargs, activation=activation)(net)

    return f

  return new_layer

class BatchNormed(object):
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def __getitem__(self, layer):
    return _batch_normed(layer, *self.args, **self.kwargs)

batch_normed = BatchNormed