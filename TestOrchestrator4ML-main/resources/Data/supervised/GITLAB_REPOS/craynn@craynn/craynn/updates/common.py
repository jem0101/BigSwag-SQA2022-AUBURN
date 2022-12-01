from .meta import Dataset

__all__ = [
  'MappedDataset', 'ZippedDataset'
]

class MappedDataset(Dataset):
  def __init__(self, dataset : Dataset, f):
    self.f = f
    self.dataset = dataset

    super(MappedDataset, self).__init__(seed=dataset.rng)

  def _get_single(self, item):
    import tensorflow as tf

    result = self._get_slice(
      slice(item, item + 1)
      if item != -1 else
      slice(item, None)
    )

    return tuple(
      tf.expand_dims(r, axis=0)
      for r in result
    )

  def _get_slice(self, item):
    result = self.f(
      *self.dataset._get_slice(item)
    )

    return result if isinstance(result, (tuple, list)) else (result, )

  def _get_sparse(self, item):
    result = self.f(
      *self.dataset._get_sparse(item)
    )

    return result if isinstance(result, (tuple, list)) else (result,)

  def get_subset(self, item):
    return self.dataset.subset(item).map(self.f)

  def size(self):
    return self.dataset.size()

  def _data(self, batch_size=1):
    from .utils import xmap

    result = xmap(
      self.f,
      self.dataset.indexed_seq(batch_size=batch_size),
    )

    if isinstance(result, (tuple, list)):
      return result
    else:
      return (result, )

  def shapes(self):
    size = len(self.dataset)
    probe = self._get_slice(slice(0, 1))

    return tuple(
      (size, ) + p.shape[1]
      for p in probe
    )


class ZippedDataset(Dataset):
  def __init__(self, first : Dataset, second : Dataset):
    self.first = first
    self.second = second

    super(ZippedDataset, self).__init__(seed=first.rng)

  def _get_single(self, item):
    return self.first._get_single(item) + self.second._get_single(item)

  def _get_slice(self, item):
    return self.first._get_slice(item) + self.second._get_slice(item)

  def _get_sparse(self, item):
    return self.first._get_sparse(item) + self.second._get_sparse(item)

  def get_subset(self, item):
    return self.first.subset(item).zip(self.second.subset(item))

  def size(self):
    return self.first.size()

  def _data(self, batch_size=1):
    return self.first.data(batch_size=batch_size) + self.second.data(batch_size=batch_size)

  def shapes(self):
    return self.first.shapes() + self.second.shapes()