import deepchem as dc
import numpy as np
import os


def test_numpy_dataset_get_shape():
  """Test that get_shape works for numpy datasets."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.NumpyDataset(X, y, w, ids)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_shape_single_shard():
  """Test that get_shape works for disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_shape_multishard():
  """Test that get_shape works for multisharded disk dataset."""
  num_datapoints = 100
  num_features = 10
  num_tasks = 10
  # Generate data
  X = np.random.rand(num_datapoints, num_features)
  y = np.random.randint(2, size=(num_datapoints, num_tasks))
  w = np.random.randint(2, size=(num_datapoints, num_tasks))
  ids = np.array(["id"] * num_datapoints)

  dataset = dc.data.DiskDataset.from_numpy(X, y, w, ids)
  # Should now have 10 shards
  dataset.reshard(shard_size=10)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == X.shape
  assert y_shape == y.shape
  assert w_shape == w.shape
  assert ids_shape == ids.shape


def test_disk_dataset_get_legacy_shape_single_shard():
  """Test that get_shape works for legacy disk dataset."""
  # This is the shape of legacy_data
  num_datapoints = 100
  num_features = 10
  num_tasks = 10

  current_dir = os.path.dirname(os.path.abspath(__file__))
  # legacy_dataset is a dataset in the legacy format kept around for testing
  # purposes.
  data_dir = os.path.join(current_dir, "legacy_dataset")
  dataset = dc.data.DiskDataset(data_dir)

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == (num_datapoints, num_features)
  assert y_shape == (num_datapoints, num_tasks)
  assert w_shape == (num_datapoints, num_tasks)
  assert ids_shape == (num_datapoints,)


def test_disk_dataset_get_legacy_shape_multishard():
  """Test that get_shape works for multisharded legacy disk dataset."""
  # This is the shape of legacy_data_reshard
  num_datapoints = 100
  num_features = 10
  num_tasks = 10

  # legacy_dataset_reshard is a sharded dataset in the legacy format kept
  # around for testing
  current_dir = os.path.dirname(os.path.abspath(__file__))
  data_dir = os.path.join(current_dir, "legacy_dataset_reshard")
  dataset = dc.data.DiskDataset(data_dir)

  # Should now have 10 shards
  assert dataset.get_number_shards() == 10

  X_shape, y_shape, w_shape, ids_shape = dataset.get_shape()
  assert X_shape == (num_datapoints, num_features)
  assert y_shape == (num_datapoints, num_tasks)
  assert w_shape == (num_datapoints, num_tasks)
  assert ids_shape == (num_datapoints,)
