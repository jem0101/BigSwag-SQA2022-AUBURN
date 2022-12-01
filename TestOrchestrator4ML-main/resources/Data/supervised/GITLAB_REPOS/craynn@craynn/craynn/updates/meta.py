import tensorflow as tf
from .utils import get_rng

__all__ = [
  'Dataset'
]

class Dataset(object):
  def __init__(self, seed=None):
    ### allows nice `dataset.subset[:100] syntax`
    self.subset = SubsetConstructor(self)
    self.rng = get_rng(seed)

  def set_seed(self, seed=None):
    self.rng = get_rng(seed)

  ### for a single item
  def _get_single(self, item):
    raise NotImplementedError()

  ### for slices
  def _get_slice(self, item):
    raise NotImplementedError()

  ### for any index arrays
  def _get_sparse(self, item):
    raise NotImplementedError()

  def get_subset(self, item):
    from .subsets import SlicedSubset, IndexedSubset
    if isinstance(item, slice):
      return SlicedSubset(self, item)
    else:
      return IndexedSubset(self, item)

  def size(self):
    raise NotImplementedError()

  def _data(self, batch_size=1):
    """
    Returns tensors held by the dataset. Must always return a tuple.
    Warning: may result in allocation of a new dataset.
    """
    raise NotImplementedError()

  def data(self, batch_size=1):
    result = self._data(batch_size=batch_size)
    return result

  def numpy(self, batch_size=1):
    return tuple(
      d.numpy()
      for d in self._data(batch_size=batch_size)
    )

  def materialize(self):
    from .variable import variable_dataset
    return variable_dataset(
      *self._data()
    )

  def shapes(self):
    raise NotImplementedError()

  def __getitem__(self, item):
    if isinstance(item, int):
      return self._get_single(item)

    elif isinstance(item, slice):
      return self._get_slice(item)

    else:
      return self._get_sparse(item)

  def __len__(self):
    return int(self.size())

  def seq(self, batch_size=1):
    from .utils import sliced_seq

    if batch_size is None:
      for i in range(len(self)):
        yield self._get_single(i)
    else:
      for indx in sliced_seq(self.size(), batch_size=batch_size):
        yield self._get_slice(indx)

  def indexed_seq(self, batch_size=1):
    from .utils import sliced_seq

    if batch_size is None:
      for i in range(len(self)):
        yield i, self._get_single(i)
    else:
      for indx in sliced_seq(self.size(), batch_size=batch_size):
        yield indx, self._get_slice(indx)

  def batch(self, batch_size=1):
    if batch_size is None:
      indx = self.rng.uniform(
        shape=(),
        dtype=tf.int32,
        minval=0,
        maxval=self.size()
      )

      return self._get_single(indx)
    else:
      indx = self.rng.uniform(
        shape=(batch_size, ),
        dtype=tf.int32,
        minval=0,
        maxval=self.size()
      )

      return self._get_sparse(indx)

  def eval(self, f=None, batch_size=1):
    if f is None:
      return self.numpy(batch_size=batch_size)
    else:
      return self.map(f).numpy(batch_size=batch_size)

  def map(self, f):
    from .common import MappedDataset
    return MappedDataset(self, f)

  def zip(self, other):
    from .common import ZippedDataset
    return ZippedDataset(self, other)

class SubsetConstructor(object):
  def __init__(self, dataset : Dataset):
    self.dataset = dataset

  def __getitem__(self, item):
    return self(item)

  def __call__(self, item):
    if isinstance(item, int):
      item = slice(item, item + 1) if item != -1 else slice(item, None)

    return self.dataset.get_subset(item)