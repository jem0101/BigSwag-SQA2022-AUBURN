import tensorflow as tf

from .. import layers
from .. import parameters
from ..subnetworks import achain

from .network_utils import get_signature, get_named_layers

__all__ = [
  'Network', 'network',
  'modify_network'
]

class Network(object):
  def __init__(self, inputs, outputs, **modes):
    self._inputs = inputs
    self._outputs = outputs
    self._modes = modes

    try:
      self.__call__.__signature__ = get_signature(inputs)
    except:
      pass

    self._named_layers = get_named_layers(self.layers())
    self._mode_cache = dict()

  def outputs(self):
    if isinstance(self._outputs, (list, tuple)):
      return self._outputs
    else:
      return (self._outputs, )

  def inputs(self):
    return self._inputs

  def find_layer(self, layer_or_name):
    if isinstance(layer_or_name, str):
      return self._named_layers[layer_or_name]
    elif isinstance(layer_or_name, layers.Layer):
      return layer_or_name
    else:
      raise Exception("%s is not a layer or a layer's name" % (layer_or_name, ))

  def find_layers(self, layers_names):
    if isinstance(layers_names, (list, tuple)):
      return tuple(
        self.find_layer(ln)
        for ln in layers_names
      )
    else:
      return self.find_layer(layers_names)

  def subnet(self, inputs=None, outputs=None):
    inputs = self._inputs if inputs is None else self.find_layers(inputs)
    inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs, )

    outputs = self._outputs if outputs is None else self.find_layers(outputs)

    return Network(inputs, outputs)

  def _as_subnet(self, *incoming):
    from ..subnetworks import subnetwork
    return subnetwork(self.inputs(), self.outputs())(*incoming)

  def _map_inputs(self, args, kwargs):
    substitutes = dict(zip(self.inputs(), args))

    for name, value in kwargs.items():
      if name in self._named_layers:
        layer = self._named_layers[name]
      else:
        raise Exception('There is no layer with name %s' % (name,))

      if layer in substitutes:
        raise Exception('Value for layer %s is provided twice, via a positional and a keyword arguments' % (name,))

      substitutes[layer] = value

    return substitutes

  @tf.function(autograph=False)
  def __call__(self, *args, **kwargs):
    is_arg_layer = [isinstance(arg, layers.Layer) for arg in args]

    if all(is_arg_layer) and len(is_arg_layer) > 0:
      if len(kwargs) != 0:
        raise Exception('Network as a SubnetworkLayer does not accept kwargs')
      return self._as_subnet(*args)

    if any(is_arg_layer) or any([ isinstance(arg, layers.Layer) for arg in kwargs.values() ]):
      raise NotImplementedError('Network can not be called on a mixture of layers and tensors yet.')

    substitutes = self._map_inputs(args, kwargs)

    try:
      return layers.get_output(self._outputs, substitutes=substitutes, **self._modes)

    except Exception as e:
      inputs_wo_substitute = [
        layer
        for layer in self.inputs()
        if layer not in substitutes
      ]

      if len(inputs_wo_substitute) > 0:
        raise ValueError('Not all inputs were provided value, this might be the cause of the error.') from e
      else:
        raise


  def mode(self, **modes):
    if len(modes) == 0:
      return self

    new_modes = self._modes.copy()
    for k, v in modes.items():
      new_modes[k] = v

    mode_key = tuple(new_modes.items())
    if mode_key not in self._mode_cache:
      self._mode_cache[mode_key] = Network(self._inputs, self._outputs, **new_modes)

    self._mode_cache[mode_key]._mode_cache = self._mode_cache

    return self._mode_cache[mode_key]


  def reset(self):
    return [
      param.reset()
      for param in self.parameters()
    ]

  def parameters(self, **properties):
    return parameters.get_all_parameters(self.outputs(), **properties)

  def variables(self, trainable=None, **properties):
    return parameters.get_all_variables(self.outputs(), trainable=trainable, **properties)

  def assign(self, variable_values, trainable=True, **properties):
    variables = self.variables(trainable=trainable, **properties)
    assert len(variables) == len(variable_values), \
      'number of variables is not the same as the number of values (%d vs %d)' % (len(variables), len(variable_values))

    for var, value in zip(variables, variable_values):
      var.assign(value)

  def description(self, short=True, **attributes):
    from craynn.info.layers import graph_description
    return graph_description(self.outputs(), short=short, inputs=self.inputs(), **attributes)

  def __str__(self):
    return self.description(short=True)

  def __repr__(self):
    return self.description(short=True)

  def total_number_of_parameters(self):
    from ..info.layers import get_total_number_of_parameters
    return get_total_number_of_parameters(self.outputs())

  def layers(self):
    return layers.get_layers(self.outputs())

  def input_shapes(self):
    return layers.get_output_shape(self.inputs())

  def output_shapes(self):
    return layers.get_output_shape(self.outputs())

  def reg_l2(self, weights=True, **properties):
    from ..regularization import apply_regularization, reg_l2

    return apply_regularization(
      self.outputs(), reg_l2(), weights=weights, **properties,
    )

  def reg_l1(self, weights=True, **properties):
    from ..regularization import apply_regularization, reg_l1

    return apply_regularization(
      self.outputs(), reg_l1(), weights=weights, **properties,
    )


def __is_shape(shape_or_layer):
  return hasattr(shape_or_layer, '__iter__') and all([ (type(s) is int or s is None) for s in shape_or_layer ])

def _get_input_layer(shape_or_layer, name=None, index=None):
  if __is_shape(shape_or_layer) :
    shape = shape_or_layer

    if name is not None:
      return layers.InputLayer(shape=shape, name=name)
    elif index is not None:
      return layers.InputLayer(shape=shape, name='input%d' % index)
    else:
      return layers.InputLayer(shape=shape, name='input')

  elif isinstance(shape_or_layer, (layers.Layer, parameters.Parameter)) :
    return shape_or_layer


def _make_network(factory, inputs, named_inputs):
  input_layers = []

  for i, input in enumerate(inputs):
    input_layers.append(_get_input_layer(input, name=None, index=i))

  for i, (name, input) in enumerate(named_inputs.items()):
    input_layers.append(_get_input_layer(input, name=name, index=i))

  explicit_names = [ layer.name for layer in input_layers if layer.name is not None ]
  assert len(set(explicit_names)) == len(explicit_names)

  outputs = factory(*input_layers)

  return Network(input_layers, outputs)


network = lambda *inputs, **named_inputs: lambda *factory: \
  _make_network(achain(*factory), inputs, named_inputs)

network.__doc__ = \
"""
Allows nice syntax:
```
  net(session)(input1, input2, ..., named_input=)(
    constructor
  )
```

or

```
  net(session)(input)(
    constructor
  )
```

for single input.
"""

def modify_network(operator, nn : Network):
  """
  Performs `craynn.layers.meta.modify_graph` of network's graph...
  See `craynn.layers.meta.modify_graph` documentation for details.
  :param operator: modification operator.
  :param nn: an instance of Network.
  :return: modified network.
  """
  import craygraph
  modified = craygraph.modify_graph(operator, nn.outputs)
  return nn.__class__(nn.inputs, modified)