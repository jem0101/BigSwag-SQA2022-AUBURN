import numpy as np

__all__ = [
  'normalize',
  'box',
  'onehot',
  'categorical_encoding',
  'categorical_to_numerical',
  'binary_encoding',
  'split'
]

def normalize(data_ref, center=True, scale=True, axis=(0, ), scale_eps=1e-3, inline=True):
  """
  (X - mean) / (std + eps)

  Statistics are computed for the first argument.

  Example:
    `data_train, data_test = normalize(data_train)(data_train, data_test)`

  Returns: transform function
  """

  mean = np.mean(data_ref, axis=axis) if center else None
  std = np.std(data_ref, axis=axis) + scale_eps if scale else None

  def transform(*data):
    if len(data) == 0:
      return tuple()

    broadcast = tuple(
      None if i in axis else slice(None, None, None)
      for i, _ in enumerate(data[0].shape)
    )
    results = data if inline else tuple([ d.copy() for d in data ])

    for d, out in zip(data, results):
      if center:
        centered = np.subtract(d, mean[broadcast], out=out)
      else:
        centered = d

      if scale:
        np.divide(centered, std[broadcast], out=out)

    return results

  return transform

def box(data_ref, inline=True, eps=1e-3):
  max = np.max(data_ref, axis=0)
  min = np.min(data_ref, axis=0)
  delta = np.maximum(max - min, eps)

  def transform(*data):
    if len(data) == 0:
      return tuple()

    broadcast = (None, ) + tuple(slice(None, None, None) for _ in data[0].shape[1:])
    results = data if inline else tuple([ d.copy() for d in data ])

    for d, out in zip(data, results):
      np.subtract(d, min[broadcast], out=out)
      np.divide(out, delta[broadcast], out=out)

    return results

  return transform


def onehot(y, n_classes=None, dtype='float32'):
  if y.dtype.kind != 'i':
    y = y.astype('int64')

  if n_classes is None:
    n_classes = np.max(y) + 1

  y_ = np.zeros(shape=(y.shape[0], n_classes), dtype=dtype)
  y_[np.arange(y.shape[0]), y] = 1

  return y_

def ceil_log2(n):
  i = 0; x = 1

  while n > x:
    x *= 2; i += 1

  return i

def categorical_encoding(*data, fixed=None):
  if fixed is None:
    encoding = dict()
  else:
    encoding = fixed.copy()

  current = 0

  for d in data:
    for item in d:
      if item not in encoding:
        while current in encoding.values():
          current += 1

        encoding[item] = current

  return encoding

def categorical_to_numerical(data, encoding=None, fixed=None, dtype='int32'):
  if encoding is None:
    encoding = categorical_encoding(data, fixed=fixed)

  result = np.ndarray(shape=(len(data),), dtype=dtype)

  for i, item in enumerate(data):
    result[i] = encoding[item]

  return result

def binary_encoding(y, dtype='float32', size=None):
  y = np.array(y)

  if size is None:
    size = size if size is not None else (np.max(y) + 1)

  n_bits = ceil_log2(size)
  power_2 = 2 ** np.arange(n_bits)[::-1]

  return np.where(
    np.mod(y[:, None] // power_2, 2) != 0,
    np.ones(shape=tuple(), dtype=dtype),
    np.zeros(shape=tuple(), dtype=dtype),
  )

def split(*data, split_ratios=0.8, seed=None):
  if len(data) == 0:
    return tuple()

  try:
    iter(split_ratios)
  except TypeError:
    split_ratios = (split_ratios, 1 - split_ratios)

  assert all([r >= 0 for r in split_ratios])

  size = len(data[0])

  split_ratios = np.array(split_ratios)
  split_sizes = np.ceil((split_ratios * size) / np.sum(split_ratios)).astype('int64')
  split_bins = np.cumsum([0] + list(split_sizes))
  split_bins[-1] = size

  rng = np.random.RandomState(seed)
  r = rng.permutation(size)

  result = list()

  for i, _ in enumerate(split_ratios):
    from_indx = split_bins[i]
    to_indx = split_bins[i + 1]
    indx = r[from_indx:to_indx]

    for d in data:
      if isinstance(d, np.ndarray):
        result.append(d[indx])
      else:
        result.append([ d[i] for i in indx ])

  return tuple(result)