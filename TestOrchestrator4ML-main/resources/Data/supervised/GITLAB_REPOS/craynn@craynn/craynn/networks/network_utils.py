from inspect import *

from craygraph import get_name

__all__ = [
  'get_signature', 'get_named_layers'
]

def get_signature(inputs):
  return Signature(
    parameters=[
      Parameter(name=input.name, kind=Parameter.POSITIONAL_OR_KEYWORD)
      for input in inputs
    ]
  )

def get_named_layers(layers):
  named_layers = dict()
  for layer in layers:
    name = get_name(layer)

    if name is None:
      continue

    if name not in named_layers:
      named_layers[name] = layer
    else:
      if isinstance(layer, layers.InputLayer) and isinstance(named_layers[name], layers.InputLayer):
        raise Exception('Collision in input names: %s' % (name, ))

  return named_layers
