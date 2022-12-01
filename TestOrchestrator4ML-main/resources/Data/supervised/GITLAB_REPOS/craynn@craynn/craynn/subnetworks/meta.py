from craygraph.graph import *

__all__ = [
  'achain', 'repeat', 'with_inputs', 'select', 'seek', 'nothing',
  'subnetwork'
]

def subnetwork(inputs, outputs):
  """
  Applies existing graph substituting inputs by a new set of incoming layers.
  This procedure make shallow copies of layers, thus all parameters are shared.
  """
  from ..layers.meta import propagate
  from copy import copy

  def model(*incoming):
    assert len(incoming) == len(inputs), 'wrong number of incoming layers'

    substitutes = dict(zip(inputs, incoming))
    def mutator(layer, layer_incoming):
      new_layer = copy(layer)

      if hasattr(new_layer, 'incoming'):
        new_layer.incoming = layer_incoming

      return new_layer

    mutated = propagate(mutator, outputs, substitutes=substitutes)
    return [ mutated[output] for output in outputs ]

  return model
