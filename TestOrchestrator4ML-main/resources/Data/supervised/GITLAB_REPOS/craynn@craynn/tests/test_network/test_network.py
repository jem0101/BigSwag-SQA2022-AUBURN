import numpy as np

def test_modes(tf):
  from craynn import network, dropout, dense

  net = network((None, 32))(
    dense(32), dropout(0.25), dense(2)
  )

  X = tf.random.normal(shape=(8, 32))
  dnet = net.mode(deterministic=True)

  assert np.all(net(X).numpy() != net(X).numpy())
  assert np.all(dnet(X).numpy() == dnet(X).numpy())