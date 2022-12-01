import tensorflow as tf
import collections

from .meta import Dataset

__all__ = [
  'VariableDataset',
  'empty_dataset', 'variable_dataset',
]

class VariableDataset(Dataset):
  def __init__(self, *variables, seed=None):
    self._variables = variables
    super(VariableDataset, self).__init__(seed=seed)

  def variables(self):
    return self._variables

  def _data(self, batch_size=1):
    return self._variables

  def _get_single(self, item):
    return tuple(
      var[item]
      for var in self._data()
    )

  def _get_slice(self, item):
    return tuple(
      var[item]
      for var in self._data()
    )

  def _get_sparse(self, item):
    return tuple(
      tf.gather(var, item, axis=0)
      for var in self._data()
    )

  def size(self):
    return tf.shape(self._variables[0])[0]

  def assign(self, *data):
    result = tuple(
      var.assign(value)
      for var, value in zip(self._variables, data)
    )
    return self

  def shapes(self):
    return tuple(
      var.shape
      for var in self._data()
    )

  def set(self, indices, *data):
    indices = tf.convert_to_tensor(indices, dtype=tf.int32)
    for var, value in zip(self._variables, data):
      var.scatter_update(tf.IndexedSlices(value, indices))

    return self

def make_variables(*shapes, dtypes=tf.float32):
  assert len(shapes) > 0

  if isinstance(dtypes, str) or isinstance(dtypes, tf.dtypes.DType):
    dtypes = [dtypes for _ in shapes]

  shapes = [
    shape_or_array.shape if hasattr(shape_or_array, 'shape') else tuple(shape_or_array)
    for shape_or_array in shapes
  ]

  general_shapes = [
    (None, ) + shape[1:]
    for shape in shapes
  ]

  initial_shapes = [
    tuple(0 if dim is None else dim for dim in shape )
    for shape in shapes
  ]

  if not isinstance(dtypes, collections.Iterable) or isinstance(dtypes, str):
    default_dtype = dtypes
    dtypes = []

    for shape_or_array in shapes:
      if hasattr(shape_or_array, 'dtype'):
        dtypes.append(shape_or_array.dtype)
      else:
        dtypes.append(default_dtype)

  variables = [
    tf.Variable(
      initial_value=tf.zeros(shape, dtype=dtype),
      dtype=dtype,
      validate_shape=False,
      shape=general_shape,
    ) for shape, general_shape, dtype in zip(initial_shapes, general_shapes, dtypes)
  ]

  return variables


def empty_dataset(*data, dtypes=tf.float32, seed=None):
  return VariableDataset(
    *make_variables(*data, dtypes=dtypes),
    seed=seed
  )


def variable_dataset(*data, seed=None):
  dataset = VariableDataset(
    *make_variables(
      *(d.shape for d in data),
      dtypes=tuple(tf.dtypes.as_dtype(d.dtype) for d in data),
    ),
    seed=seed
  )
  dataset.assign(*data)
  return dataset