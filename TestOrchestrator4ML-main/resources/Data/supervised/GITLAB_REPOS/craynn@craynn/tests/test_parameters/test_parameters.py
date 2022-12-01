import numpy as np
from craynn.parameters import *

def test_parameters(tf):
  W = ones_init(trainable=False)(shape=(32, 32), trainable=True)
  assert not W.properties('trainable')

  W = ones_init(trainable=True)(shape=(32, 32), trainable=False)
  assert W.properties('trainable')

  W = ones_init()(shape=(32, 32), trainable=True)
  assert W.properties('trainable')

  W = ones_init()(shape=(32, 32), trainable=False)
  assert not W.properties('trainable')

  assert W().shape == (32, 32)
  assert np.allclose(W().numpy(), np.ones(shape=(32, 32)))

  W = glorot_normal_init(gain=1.0)(shape=(32, 32), trainable=True)
  assert W().shape == (32, 32)

  W1 = W().numpy()
  W.reset()
  W2 = W().numpy()

  assert np.any(W1 != W2)

  W = decomposition(n=5, w1=ones_init(), w2=ones_init())(shape=(32, 32))

  assert W.properties('composite')
  assert not W.w1.properties('composite')
  assert not W.w2.properties('composite')

  assert W().shape == (32, 32)
  w1, w2 = W.incoming()
  assert np.allclose(w1().numpy(), np.ones(shape=(32, 5)))
  assert np.allclose(w2().numpy(), np.ones(shape=(5, 32)))
  assert len(W.variables()) == 0
  assert len(get_all_variables(W)) == 2

  W = decomposition(n=5)(shape=(32, 32))

  W1 = W().numpy()
  W.reset()
  W2 = W().numpy()

  assert np.any(W1 != W2)

  W = decomposition(n=5, w1=ones_init(composite=True, flag=False), w2=ones_init(composite=True, flag=True))(shape=(32, 32))
  assert W.properties('composite')
  assert W.w1.properties('composite')
  assert W.w2.properties('composite')

  assert not W.properties('flag')
  assert not W.w1.properties('flag')
  assert W.w2.properties('flag')

  W = const_init(1)((32, 15))
  assert W().shape == (32, 15)
  assert np.all(np.abs(W().numpy() - 1) < 1e-6)

  W = const_init(value=1)((32, 15))
  assert W().shape == (32, 15)
  assert np.all(np.abs(W().numpy() - 1) < 1e-6)

  W = const_init(value=1)(shape=(32, 15))
  assert W().shape == (32, 15)
  assert np.all(np.abs(W().numpy() - 1) < 1e-6)

  W = const_init(value=1, prop=True)(shape=(32, 15), prop=False)
  assert W().shape == (32, 15)
  assert np.all(np.abs(W().numpy() - 1) < 1e-6)
  assert W.properties('prop')

  W = unbound_parameter()(shape=(32, 32))
  assert W.properties('unbound')