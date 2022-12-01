import time
import numpy as np

def test_resize(tf):
  from craynn import empty_dynamic_dataset, variable_dataset

  N = 1024
  X_DIM = 32

  start_time = time.time()
  dataset = empty_dynamic_dataset((None, X_DIM), (None, ), capacity=N, dtypes=tf.float32)
  for i in range(N):
    dataset.append(
      np.repeat(np.float32(i), repeats=X_DIM).reshape(1, X_DIM),
      np.array([N - i - 1], dtype='float32')
    )

  X, y = dataset.data()
  end_time = time.time()

  assert np.allclose(X, np.arange(N).astype('float32')[:, None])
  assert np.allclose(y, np.arange(N).astype('float32')[::-1])

  per_iteration = (end_time - start_time) / N * 1000
  print('Append: %.3lf milliseconds per iteration' % (per_iteration, ))

  @tf.function(autograph=False)
  def batch():
    X, y = dataset.batch(32)
    return tf.reduce_sum(X - y[:, None])

  start_time = time.time()
  for i in range(N):
    _ = batch()
  end_time = time.time()

  print(batch())
  print(batch())
  print(batch())
  print(batch())

  per_iteration = (end_time - start_time) / N * 1000
  print('Batch: %.3lf milliseconds per iteration' % (per_iteration,))

  const_dataset = variable_dataset(*dataset.numpy())

  @tf.function(autograph=False)
  def batch_variable():
    X, y = const_dataset.batch(32)
    return tf.reduce_sum(X - y[:, None])

  start_time = time.time()
  for i in range(N):
    _ = batch_variable()
  end_time = time.time()

  per_iteration = (end_time - start_time) / N * 1000
  print('Variable batch: %.3lf milliseconds per iteration' % (per_iteration,))


def test_append(tf):
  from craynn import empty_dynamic_dataset

  get_data = lambda s, e: 10 * np.arange(s, e)[:, None] + np.arange(0, 10)[None, :]

  dataset = empty_dynamic_dataset((None, 10), capacity=100, dtypes=tf.float32)

  for i in range(1, 9, 2):
    dataset.append(get_data(i, i + 2))

    assert dataset.capacity().numpy() == 100
    assert dataset.size().numpy() == i + 1

    for _ in range(32):
      batch, = dataset.batch(10)
      assert np.all(batch >= 10)
      assert np.all(batch < 10 * (i + 2))

    assert np.min(dataset.data()) == 10
    assert np.max(dataset.data()) == 10 * (i + 2) - 1
    assert np.unique(dataset.data()).shape[0] == 10 * (i + 1)

  assert np.allclose(dataset.data()[0], get_data(1, 9))

def test_regression(tf):
  from craynn import empty_dynamic_dataset
  from craynn import network, dense, flatten, nonlinearities
  from craynn import objectives

  from crayopt import tf_updates

  N_SAMPLES = 1024
  N_DIMS = 2
  N_STEPS_PER_UPDATE = 8

  def get_data(size, n_dims):
    X = np.random.uniform(low=-1, high=1, size=(size, n_dims)).astype('float32')
    return X, np.sum(X ** 2, axis=1)

  X, y = get_data(N_SAMPLES, N_DIMS)
  indx = np.argsort(X[:, 0])
  X, y = X[indx], y[indx]

  net = network((None, N_DIMS))(
    dense(64, activation=nonlinearities.softplus()),
    dense(32, activation=nonlinearities.softplus()),
    dense(1, activation=nonlinearities.linear()),
    flatten(1)
  )

  dataset = empty_dynamic_dataset((None, N_DIMS), (None, ), capacity=X.shape[0], dtypes=tf.float32)
  optimizer = tf_updates.adam(learning_rate=1e-3)(net.variables(trainable=True))

  loss_f = objectives.mse()

  @tf.function(autograph=False)
  def step():
    X_batch, y_batch = dataset.batch(32)

    with optimizer:
      loss = loss_f(y_batch, net(X_batch))
      return optimizer(loss)

  losses = np.zeros(shape=(X.shape[0], N_STEPS_PER_UPDATE))

  for i in range(X.shape[0]):
    dataset.append(X[i:(i + 1)], y[i:(i + 1)])

    for j in range(N_STEPS_PER_UPDATE):
      losses[i, j] = step()

  X_test, y_test = get_data(N_SAMPLES, N_DIMS)

  baseline_p = np.mean(y)
  baseline_MSE = np.mean((y_test - baseline_p) ** 2)
  MSE = np.mean((y_test - net(X_test)) ** 2)

  r_score = 1 - MSE / baseline_MSE
  print('R-score: %.3lf' % (r_score, ))

  if r_score < 0.5:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 6))
    plt.plot(np.mean(losses, axis=1), label='dynamic regression (R-score = %.2lf)' % (r_score, ))
    plt.savefig('dynamic-regression.png')
    plt.close()

    raise Exception('R-score is too high, dynamic dataset potentially does not work as intended')
