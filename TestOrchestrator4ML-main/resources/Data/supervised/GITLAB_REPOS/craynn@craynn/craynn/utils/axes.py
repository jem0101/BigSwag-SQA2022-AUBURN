__all__ = [
  'normalize_axis',
  'gsum'
]

def normalize_axis(tensor_or_dim, axis):
  if isinstance(tensor_or_dim, int):
    dim = tensor_or_dim
  elif isinstance(tensor_or_dim, tuple):
    dim = len(tensor_or_dim)
  else:
    dim = len(tensor_or_dim.shape)

  if isinstance(axis, int):
    return axis % dim
  else:
    return tuple(a % dim for a in axis)

def gsum(xs):
  acc = 0
  for x in xs:
    if x is None:
      return None
    else:
      acc += x

  return acc