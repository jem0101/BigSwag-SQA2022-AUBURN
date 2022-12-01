__all__ = [
  'Nonlinearity', 'nonlinearity_from'
]

class Nonlinearity(object):
  def __init__(self, nonlinearity, hyperparameters):
    self.nonlinearity = nonlinearity
    self.hyperparameters = hyperparameters
    
    super(Nonlinearity, self).__init__()

  def __call__(self, x):
    return self.nonlinearity(x)

  def __str__(self):
    return '%s(%s)' % (
      self.__class__.__name__,
      ', '.join([
        '%s=%s' % (k, v) for k, v in self.hyperparameters.items()
      ])
    )

def nonlinearity_from(f):
  import inspect

  signature = inspect.signature(f)
  name_parameter, = [
    signature.parameters[p] for p in signature.parameters if p == 'name'
  ]

  name = name_parameter.default

  def __init__(self, *args, **kwargs):
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()

    g = f(*bound.args, **bound.kwargs)

    hyperparameters = bound.arguments.copy()
    hyperparameters.pop('name', None)

    Nonlinearity.__init__(self, g, hyperparameters)

  clazz = type(
    name,
    (Nonlinearity, ),
    dict(__init__ = __init__)
  )

  return clazz