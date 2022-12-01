from craygraph import Node, NodeModel
from craygraph import CarryingExpression, apply_with_kwargs
from craygraph import get_incoming, propagate, dynamic_reduce_graph, get_nodes, graph_reducer

__all__ = [
  'Layer', 'LayerModel', 'model_from',

  'InputLayer',

  'get_incoming',

  'get_output',
  'get_all_outputs',

  'get_output_shape',
  'get_all_output_shapes',

  'get_layers',
  'get_inputs',

  'model_selector',
]


class Layer(Node):
  def __init__(self, *incoming, parameters=(), name=None, shape=None):
    self._parameters = parameters
    self._incoming_layers = incoming

    super(Layer, self).__init__(*(parameters + incoming), name=name)

    if shape is None:
      incoming_shapes = get_output_shape(self.incoming())
      self._shape = self.get_output_shape_for(*incoming_shapes)
    else:
      self._shape = shape

  def incoming_layers(self):
    return self._incoming_layers

  def parameters(self):
    return self._parameters

  def shape(self):
    return self._shape


  def get_output_shape_for(self, *input_shapes, **kwargs):
    raise NotImplementedError()

  def get_output_for(self, *inputs, **kwargs):
    raise NotImplementedError()

class LayerModel(NodeModel):
  def __call__(self, *args, **kwargs):
    raise NotImplementedError()


class InputLayer(Layer):
  def __init__(self, shape, name=None):
    super(InputLayer, self).__init__(name=name, shape=shape)

  def get_output_shape_for(self):
    return self._shape

  def get_output_for(self, *inputs, **kwargs):
    raise ValueError('Input was asked to return a value, this should not have happened.')

def model_from(clazz, incoming_args=None):
  import inspect
  signature = inspect.signature(clazz)

  if incoming_args is None:
    if 'incoming' in signature.parameters.keys():
      carried = ('incoming', )
    else:
      carried = tuple()
  else:
    carried = incoming_args

  return CarryingExpression(clazz, carried=carried, base_constructor_class=LayerModel)

get_output = graph_reducer(lambda layer: layer.get_output_for, strict=False)

def get_output_shape(layer):
  def cached_shape(node):
    shape_f = getattr(node, 'shape', lambda: None)
    if callable(shape_f):
      return shape_f()
    else:
      return None

  def incoming(node):
    result = cached_shape(node)

    if result is None:
      return get_incoming(node), None
    else:
      return tuple(), result

  def operator(node, args, cached):
    if cached is not None:
      return cached
    else:
      return node.get_output_shape_for(*args)

  cached = cached_shape(layer)

  if cached is None:
    return dynamic_reduce_graph(operator, layer, incoming=incoming)
  else:
    return cached

def get_all_outputs(layer_or_layers, substitutes, **modes):
  operator = lambda layer, inputs: layer.get_output_for(*inputs, **modes)

  if isinstance(layer_or_layers, Layer):
    layers = [layer_or_layers]
  else:
    layers = layer_or_layers

  return propagate(operator, layers, substitutes)

get_all_output_shapes = lambda layers, substitutes=None, **kwargs: propagate(
  lambda layer, args: apply_with_kwargs(layer.get_output_shape_for, *args, **kwargs),
  layers, substitutes=substitutes
)

get_layers = lambda layers_or_layer: [
  node
  for node in get_nodes(layers_or_layer)
  if isinstance(node, Layer)
]

get_inputs = lambda layers_or_layer: [
  layer
  for layer in get_layers(layers_or_layer)
  if isinstance(layer, Layer) and len(get_incoming(layer)) == 0
]


def model_selector(criterion):
  """Decorator, changes signature and inserts checks into a layer model selector.

  This is a wrapper which inserts a common procedures for a selector:
  - signature checks for each model (must all be the same);
  - binding of model parameters;
  - replacement of selector signature by models' shared signature.

  Model selector is a meta layer model that selects a particular layer model from a provided list
  based on properties of incoming layer, i.e. defers selection of a model until network construction.

  Type of the selector: `list of models` -> `incoming layer` -> `layer`.

  Parameters
  ----------
  criterion : Selector
    selector to modify. This function can assume that models have the same signature (i.e. accept the same parameters).

  Returns
  -------
  Selector
    Selector with changed signature and validity checks.

  Examples
  -------
  Selecting model with proper dimensionality for convolutional layer
  based on dimensionality of the incoming layer:

  >>> @model_selector
  >>> def dimensionality_selector(models):
  >>>   def common_model(incoming):
  >>>     ndim = get_output_shape(incoming) - 2
  >>>     return models[ndim]
  >>>   return common_model
  """

  def selector(models):
    from inspect import signature, Signature, Parameter
    assert len(models) > 0

    models_signatures = [
      signature(model) for model in models
      if model is not None
    ]

    if len(set(models_signatures)) != 1:
      pretty_signatures = '\n  '.join([ str(signature) for signature in set(models_signatures)])
      raise ValueError('All models must have the same signature, got:%s' % pretty_signatures)

    common_signature = models_signatures[0]
    pretty_parameters = [Parameter('self', Parameter.POSITIONAL_ONLY)]
    pretty_parameters.extend(common_signature.parameters.values())
    pretty_signature = Signature(parameters=pretty_parameters, return_annotation=common_signature.return_annotation)

    bound_criterion = criterion(models)

    def __init__(self, *args, **kwargs):
      self.args = args
      self.kwargs = kwargs

      common_signature.bind(*args, **kwargs)

    def __call__(self, *incoming):
      selected_model = bound_criterion(*incoming)
      if selected_model is None:
        raise ValueError('Invalid incoming layer!')

      return selected_model(*self.args, **self.kwargs)(*incoming)

    __init__.__signature__ = pretty_signature

    model = type(
      '%s' % (getattr(criterion, '__name__', 'model_selector'), ),
      (LayerModel, ),
      dict(
        __init__ = __init__,
        __call__ = __call__
      )
    )

    return model

  return selector
