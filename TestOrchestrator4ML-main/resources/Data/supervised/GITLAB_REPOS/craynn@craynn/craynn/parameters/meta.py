import tensorflow as tf
from craygraph import get_nodes, Node, CarryingExpression

from .parameter_utils import combine_properties

__all__ = [
  'Parameter', 'ParameterModel',

  'ConstantParameter', 'constant_parameter',
  'FreeParameter', 'as_free_parameter',

  'UnboundParameter', 'unbound_parameter',
  'BoundParameter', 'bound_parameter',

  'parameter_model',

  'check_properties',
  'get_parameters', 'get_all_parameters',
  'get_variables', 'get_all_variables',

  'ParameterCloneMachine', 'shared_parameter'
]

class Parameter(Node):
  def __init__(self, *incoming, variables=(), shape=(), name=None, **properties):
    self._shape = shape
    self._properties = properties
    self._variables = variables
  
    super(Parameter, self).__init__(*incoming, name=name)

  def properties(self, item=None):
    if item is None:
      return self._properties
    else:
      return self._properties.get(item, False)

  def get_output_for(self, *incoming, **modes):
    raise NotImplementedError()

  def __call__(self, **modes):
    from ..layers import get_output

    incoming = get_output(self.incoming())
    return self.get_output_for(*incoming, **modes)

  def get_output_shape_for(self, *input_shapes):
    return self._shape

  def shape(self):
    return self._shape

  def variables(self):
    """
    Returns variables held by the parameter. A variable is held by a parameter if it is owned by the parameter.
    Must not include variables of the dependencies.

    A variable shared between multiple parameters must be owned by only one of the parameters.
    See `shared_parameter`.

    :return: list of `tf.Variable`
    """
    return self._variables

  def reset(self):
    for param in self.incoming():
      getattr(param, 'reset', lambda: None)()

  def __str__(self):
    name = self.__class__.__name__ if self._name is None else self._name
    shape = 'x'.join([ '%d' % (s, ) for s in self.shape() ])
    props = [('shape', shape)] + list(self._properties.items())

    return '%s (%s)' % (
      name,
      ', '.join([ '%s=%s' % (k, v) for k, v in props ])
    )

  def __repr__(self):
    return str(self)

  def transform(self, f):
    """
    Returns a TransformedParameter that applies function `f` to the parameter's value.
    :param f: transformation function;
    :return: an instance of TransformedParameter.
    """
    return TransformedParameter(self, f)


class ParameterModel(object):
  pass


def parameter_model(clazz, **default_properties):
  return CarryingExpression(
    original=clazz,
    carried=('shape', 'properties', 'name'),
    duplicated=dict(
      properties=combine_properties,
      name=lambda custom_name, default_name: default_name if custom_name is None else custom_name
    ),
    base_constructor_class=ParameterModel,
    defaults=default_properties
  ).with_defaults(name=None)


class TransformedParameter(Parameter):
  def __init__(self, incoming : Parameter, f, name=None):
    super(TransformedParameter, self).__init__(incoming, shape=incoming.shape(), name=name)
    self.f = f

  def get_output_for(self, incoming):
    return self.f(incoming)

  def get_output_shape_for(self, incoming_shape):
    return incoming_shape

  def properties(self, item=None):
    incoming, = self.incoming()
    return incoming.properties(item=item)


class ConstantParameter(Parameter):
  def __init__(self, value, name=None, **properties):
    dtype = tf.dtypes.as_dtype(getattr(value, 'dtype', tf.float32))
    self._value = tf.constant(value, dtype=dtype, name=name)

    super(ConstantParameter, self).__init__(shape=value.shape, name=name, **properties)

  def __call__(self):
    return self._value

  def get_output_for(self,):
    return self._value

  def reset(self):
    pass

  def transform(self, f):
    value = f(self._value)
    return ConstantParameter(value, properties=self.properties(), name=self.name())

constant_parameter = CarryingExpression(
  ConstantParameter,
  carried=('properties', 'name'),
  duplicated=dict(
    properties=combine_properties,
    name=lambda custom_name, default_name: default_name if custom_name is None else custom_name,
  ),
  base_constructor_class=ParameterModel
)()


class FreeParameter(Parameter):
  ### everything for pickle
  def __init__(self, initializer, initializer_arguments, name=None, **properties):
    self._initializer = initializer
    self._initializer_arguments = initializer_arguments

    initial = initializer(**self._initializer_arguments)
    self._value = tf.Variable(
      initial_value=initial,
      name=name,
      dtype=initial.dtype,
      trainable=properties.get('trainable', False)
    )

    super(FreeParameter, self).__init__(
      shape=initializer_arguments['shape'],
      name=name,
      variables=(self._value, ),
      **properties
    )

  def get_output_for(self):
    return self._value

  def reset(self):
    return self._value.assign(
      self._initializer(**self._initializer_arguments)
    )

  def assign(self, value):
    return self._value.assign(value)


def as_free_parameter(f, name=None, **default_properties):
  import inspect

  capitalize = lambda s: s[0].upper() + s[1:]

  if name is None:
    clazz_name = '%sInit' % (capitalize(
      f.__name__[1:]
      if f.__name__.startswith('_') else
      capitalize(f.__name__)
    ))
  else:
    clazz_name = name

  original_signature = inspect.signature(f)

  assert 'properties' not in original_signature.parameters, \
    'decorated function has a conflicting parameter (properties)'

  assert 'shape' in original_signature.parameters, \
    'decorated function does not have shape parameter'

  parameters = list(original_signature.parameters.values())
  if 'name' not in original_signature.parameters:
    parameters.append(inspect.Parameter('name', kind=inspect.Parameter.KEYWORD_ONLY, default=None))

  for k, v in default_properties.items():
    assert k not in original_signature.parameters, \
      "default property is present in the original function's signature."
    parameters.append(inspect.Parameter(k, kind=inspect.Parameter.KEYWORD_ONLY, default=v))

  parameters.append(inspect.Parameter('properties', kind=inspect.Parameter.VAR_KEYWORD))

  clazz_signature = inspect.Signature(parameters=parameters, return_annotation=original_signature.return_annotation)
  apparent_clazz_signature = inspect.Signature(
    parameters=[inspect.Parameter('self', kind=inspect.Parameter.POSITIONAL_ONLY)] + parameters,
    return_annotation=original_signature.return_annotation
  )

  def __init__(self, *args, **kwargs):
    arguments = clazz_signature.bind(*args, **kwargs)
    arguments.apply_defaults()

    properties = arguments.arguments.pop('properties')

    _name = arguments.arguments.get('name', None)

    FreeParameter.__init__(
      self,
      initializer=f,
      initializer_arguments=arguments.arguments,
      name=_name,
      **properties
    )

  __init__.__signature__ = apparent_clazz_signature
  clazz = type(clazz_name, (FreeParameter, ), dict(__init__=__init__))

  return clazz

class UnboundParameter(Parameter):
  def __init__(self, shape, dtype='float32', name=None, **properties):
    super(UnboundParameter, self).__init__(shape, name=name, **properties)
    self.dtype = dtype

  def get_output_for(self, ):
    raise NotImplementedError('Unbound parameters are meant to be substituted.')

  def __str__(self):
    return '%s: %s' % (
      self.__class__.__name__ if self.name is None else self.name,
      str(self._shape)
    )

unbound_parameter = parameter_model(UnboundParameter, unbound=True)()

class BoundParameter(Parameter):
  def __init__(self, incoming, shape, name=None, **properties):
    super(BoundParameter, self).__init__(incoming, shape=shape, name=name, **properties)

  def get_output_for(self, incoming):
    return incoming

bound_parameter = parameter_model(BoundParameter)()


def check_properties(**properties):
  effective_properties = tuple(
    (k, v)
    for k, v in properties.items()
    if v is not None
  )

  def predicate(param):
    props = getattr(param, 'properties', dict)()

    return all([
      (props.get(k, False) == v)
      for k, v in effective_properties
    ])

  return predicate

def get_all_parameters(node, **properties):
  """
  Get all parameters that satisfy all `properties` from the subgraph defined by `node`.

  A parameter satisfies a property `prop = value` if:
    - value is None;
    - the parameter has property `prop` and its value equals to `value` or
    - the parameter lacks property `prop` and `value = False`.

  Note, that `prop = None` matches all parameters, this is done to
  place commonly used properties to default arguments and enable autocomplete for them.

  :param node: an instance of Node (e.g. Layer or Parameter), a list or a tuple of nodes.
  :param properties: properties to select by.
  :return: list of all parameters that satisfy `properties`
  """
  check_props = check_properties(**properties)

  return [
    node
    for node in get_nodes(node)
    if isinstance(node, Parameter)
    if check_props(node)
  ]

def get_parameters(node, **properties):
  """
  Get parameters of the node. Not, that unlike `get_all_parameters`,
  this function returns only parameters of `node` and does not inspects parameters of incoming node.
  In particular, if node depends on a parameter that, in its turn, depends on another parameter,
  only the first parameter will be returned.

  :param node: an instance of Node (e.g. Layer or Parameter).
  :param properties: properties to select by.
  :return: list of all parameters that satisfy `properties`
  """
  check_props = check_properties(**properties)

  return [
    param
    for param in getattr(node, 'parameters', tuple)()
    if check_props(param)
  ]

def get_all_variables(layer, **properties):
  """
    Return all variables from parameters that satisfy `properties`.
    Note that variables themselves do not have properties, but parameters have.

    Note, that this function assumes empty property dict if a node does not have `properties` attribute, thus,
    layers that have variables also might be included in the output, e.g., if no filters are to apply.

    :param layer: an instance of Layer, a list or a tuple of layers.
    :param properties: properties to select parameters by.
    """
  check = check_properties(**properties)

  return tuple(
    var

    for node in get_nodes(layer)
    if check(node)

    for var in getattr(node, 'variables', tuple)()
  )

def get_variables(layer, **properties):
  return tuple(
    var
    for parameter in get_parameters(layer, **properties)
    for var in getattr(parameter, 'variables', tuple)()
  )


class ParameterCloneMachine(object):
  def __init__(self, parameter_constructor):
    self.parameter_constructor = parameter_constructor
    self.parameter = None
    self._shape = None

  def __call__(self, shape, name=None, **additional_properties):
    if self.parameter is None:
      self.parameter = self.parameter_constructor(shape, name, **additional_properties)
      self._shape = shape
      return self.parameter

    else:
      assert shape == self._shape, 'Can not clone parameter for different shape.'
      return self.parameter

shared_parameter = ParameterCloneMachine

