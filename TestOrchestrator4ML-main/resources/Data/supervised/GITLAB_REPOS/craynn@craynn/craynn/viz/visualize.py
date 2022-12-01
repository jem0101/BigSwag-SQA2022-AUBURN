from craygraph import visualize

from .. import layers
from .. import parameters
from ..info.layers import get_number_of_parameters

__all__ = [
  'draw_to_file',
  'draw_to_notebook',

  'all_nodes', 'only_layers'
]

def map_inputs(layer, incoming):
  import inspect
  signature = inspect.signature(layer.get_output_for)

  if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in signature.parameters.values()):
    return incoming
  else:
    assert len(signature.parameters) >= len(incoming)
    return dict(zip(signature.parameters, incoming))

def all_nodes(node, args):
  incoming = node.incoming()

  if len([arg for arg in args if arg is not None]) > 1:
    return map_inputs(node, incoming)
  else:
    return incoming

def only_layers(node, args):
  if any(arg is not None for arg in args):
    return all_nodes(node, args)

  if isinstance(node, parameters.Parameter):
    return None
  else:
    return all_nodes(node, args)

_color_set = (
  ### blue
  (('conv', ), ('#a6cee3', '#1f78b4')),
  ### green
  (('dense', layers.DenseLayer), ('#b2df8a', '#33a02c')),
  ### red
  (('input', layers.InputLayer), ('#fb9a99', '#e31a1c')),
  ### orange
  (('pool', ), ('#fdbf6f', '#ff7f00')),
  ### magenta
  (('recurrent', ), ('#cab2d6', '#6a3d9a')),
  ### yellow/brown
  (('', ), ('#b15928', '#ffff99')),
)

def _stable_hash(data):
  from hashlib import blake2b
  return int.from_bytes(blake2b(data.encode('utf-16be'), digest_size=8).digest(), byteorder='big')

def layer_color(color_set=_color_set):
  def get_color(layer):
    layer_class = layer.__class__
    layer_type = layer_class.__name__.lower()

    hashed = _stable_hash(layer_class.__name__)

    for condition, colors in color_set:
      for cond in condition:
        if isinstance(cond, type):
          if issubclass(layer_class, cond):
            j = hashed % len(colors)
            return colors[j]

        elif isinstance(cond, str):
          if cond in layer_type:
            j = hashed % len(colors)
            return colors[j]

        else:
          raise TypeError('Condition %s is not understood' % (cond, ))

    _, colors = color_set[-1]
    j = hashed % len(colors)
    return colors[j]

  return get_color

def viz_params(**kwargs):
  def f(layer, _):
    param_info = get_number_of_parameters(layer, **kwargs).items()
    if len(param_info) > 0:
      return ','.join([ '%s: %d' % (k.name, v) for k, v in param_info ])
    else:
      return None

def viz_all_params(**properties):
  def number_of_params(layer):
    n = get_number_of_parameters(layer)

    if n > 0:
      return '#params: %d' % n
    else:
      return None

  return number_of_params

def viz_name(force_class_name=False, remove_layer_from_class_name=True, kernel_info=True):
  def name(layer):
    layer_class = layer.__class__.__name__.split('.')[-1]

    if remove_layer_from_class_name:
      layer_class = layer_class.replace('Layer', '')
      layer_class = layer_class.replace('layer', '')

    kernel_info = []
    if hasattr(layer, 'kernel_size'):
      kernel_info.append(
        'x'.join('%d' % k for k in layer.kernel_size)
      )

    if hasattr(layer, 'stride'):
      if any(s != 1 for s in layer.stride):
        kernel_info.append(
          'stride=%s' % (
            'x'.join('%d' % k for k in layer.stride),
          )
        )

    if hasattr(layer, 'padding'):
      if layer.padding != 'valid':
        kernel_info.append('pad=%s' % layer.padding)

    if layer.name() is None or force_class_name:
      result = layer_class
    else:
      result = '%s : %s' % (layer.name(), layer_class)

    if kernel_info:
      return '%s %s' % (result, ', '.join(kernel_info))
    else:
      return result

  return name

def viz_output_shape():
  def output_shape(layer):
    shape = layers.get_output_shape(layer)
    return str(shape)

  return output_shape

def viz_activation():
  def activation(layer):
    return getattr(layer, 'activation', None)

  return activation

default_display_properties = dict(
  _name_ = viz_name(),
  _activation_ = viz_activation(),
  _params_ = viz_all_params(),
  _shape_ = viz_output_shape(),
  __style__ = 'filled',
  __fillcolor__ = layer_color(color_set=_color_set),
  __color__ = 'black',
  __shape__ = 'box'
)

def draw_to_file(path, network, selector=only_layers, **properties):
  props = default_display_properties.copy()
  props.update(properties)
  return visualize.draw_to_file(path, network, selector=selector, **props)

def draw_to_notebook(network, selector=only_layers, **properties):
  props = default_display_properties.copy()
  props.update(properties)
  return visualize.draw_to_notebook(network, selector=selector, **props)