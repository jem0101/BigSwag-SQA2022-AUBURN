import numpy as np

def test_clone(tf):
  from craynn.parameters import shared_parameter, glorot_uniform_init

  W = shared_parameter(glorot_uniform_init(gain=5))

  W1 = W(shape=(10, 15))

  try:
    W_ = W(shape=(9, 15))
  except AssertionError:
    pass
  else:
    raise Exception('Exception should be raised!')


  W2 = W(shape=(10, 15))
  W3 = W(shape=(10, 15))

  assert np.all(W1().numpy() == W2().numpy())
  assert np.all(W2().numpy() == W3().numpy())
  assert np.all(W1().numpy() == W3().numpy())

  var, = W1.variables()
  var.assign_add(tf.random.uniform(shape=(10, 15)))

  assert np.all(W1().numpy() == W2().numpy())
  assert np.all(W2().numpy() == W3().numpy())
  assert np.all(W1().numpy() == W3().numpy())

  assert len(W1.variables()) == 1
  assert len(W2.variables()) == 1
  assert len(W3.variables()) == 1

