import numpy as np

def test_variable_dataset(tf):
  from craynn import variable_dataset

  data = np.repeat(np.arange(12), 5, axis=0).reshape((12, 5)).astype('float32')
  dataset = variable_dataset(data)

  def check_batch(f, min, max):
    for i in range(128):
      values, = f()
      assert np.allclose(np.min(values, axis=1), np.max(values, axis=1))
      assert np.min(values) >= min and np.max(values) <= max

  print(dataset)
  print(dataset.subset[:])
  print(dataset.subset[1:])

  check_batch(lambda : dataset.batch(32), 0, 11)
  check_batch(lambda : dataset.subset[:].batch(32), 0, 11)
  check_batch(lambda : dataset.subset[1:].batch(32), 1, 11)
  check_batch(lambda : dataset.subset[:-1].batch(32), 0, 10)
  check_batch(lambda : dataset.subset[2:-2].batch(32), 2, 9)

  dataset.assign(
    np.repeat(np.arange(9), 5, axis=0).reshape((9, 5)).astype('float32')
  )

  check_batch(lambda : dataset.batch(32), 0, 8)
  check_batch(lambda : dataset.subset[:].batch(32), 0, 8)
  check_batch(lambda : dataset.subset[1:].batch(32), 1, 8)
  check_batch(lambda : dataset.subset[:-1].batch(32), 0, 7)
  check_batch(lambda : dataset.subset[2:-2].batch(32), 2, 6)

def test_operations(tf):
  from craynn import variable_dataset

  data1 = np.arange(120).reshape((12, 5, 2)).astype('float32')
  data2 = (119 - np.arange(120)).reshape((12, 5, 2)).astype('float32')

  d1 = variable_dataset(data1)
  d2 = variable_dataset(data2)

  assert np.allclose(
    d1.map(lambda x: x + 1).eval(lambda x: tf.reduce_sum(x, axis=1), batch_size=7),
    np.sum((data1 + 1), axis=1)
  )

  r1 = d1.map(lambda x: x + 1).map(lambda x: x - 1).subset[-10:10].zip(
    d2.subset[tf.range(2, 10)].map(lambda x: tf.math.log1p(x))
  ).map(lambda x, y: x + y).eval(lambda x: tf.reduce_sum(x, axis=1), batch_size=7)

  r2 = np.sum(data1[2:10] + np.log1p(data2[2:10]), axis=1)

  print()
  print(r1)
  print(r2)

  assert np.allclose(r1, r2)

def test_singular(tf):
  from craynn import variable_dataset

  data1 = np.arange(120).reshape((12, 5, 2)).astype('float32')
  data2 = (119 - np.arange(120)).reshape((12, 5, 2)).astype('float32')

  d1 = variable_dataset(data1, data2)
  b1, = d1.map(lambda x, y: x + 1).batch(32)
  assert b1.shape == (32, 5, 2)

  b1, b2 = d1.map(lambda x, y: (x + y, y)).eval()
  assert b1.shape == (12, 5, 2)
  assert b2.shape == (12, 5, 2)

  b1, b2 = d1.map(lambda x, y: x + y).map(lambda x: (x - 1, x + 1)).eval()
  assert b1.shape == (12, 5, 2)
  assert b2.shape == (12, 5, 2)

  d2 = variable_dataset(data1)
  b1, b2 = d2.map(lambda x: (x - 1, x + 2)).eval()
  assert b1.shape == (12, 5, 2)
  assert b2.shape == (12, 5, 2)

  b1, = d2.map(lambda x: (x - 1, x + 2)).map(lambda x, y: x + y).eval()
  print(b1)
  assert b1.shape == (12, 5, 2)

def get_data(size=1024):
  X1 = np.random.normal(size=(size, 32)).astype('float32')
  X2 = np.random.normal(size=(size, 2)).astype('float32')
  X3 = np.random.normal(size=(size, 7)).astype('float32')

  return X1, X2, X3

def check_batches(d1, d2, d3):
  for _ in range(32):
    b1 = d1.batch(32)
    b2 = d2.batch(32)
    b3 = d3.batch(32)

    assert all(
      np.allclose(x1, x2)
      for x1, x2 in zip(b1, b2)
    )

    assert any(
      not np.allclose(x1, x3)
      for x1, x3 in zip(b1, b3)
    )

def test_seed(tf):
  from craynn import variable_dataset
  X1, X2, X3 = get_data(1024)
  d1 = variable_dataset(X1, X2, X3, seed=111)
  d2 = variable_dataset(X1, X2, X3, seed=111)
  d3 = variable_dataset(X1, X2, X3, seed=112)
  check_batches(d1, d2, d3)


def test_set_seed(tf):
  from craynn import variable_dataset
  X1, X2, X3 = get_data(1024)
  d1 = variable_dataset(X1, X2, X3, seed=111)
  d2 = variable_dataset(X1, X2, X3, seed=112)
  d3 = variable_dataset(X1, X2, X3, seed=111)

  d3.set_seed(113)
  d2.set_seed(111)

  check_batches(d1, d2, d3)