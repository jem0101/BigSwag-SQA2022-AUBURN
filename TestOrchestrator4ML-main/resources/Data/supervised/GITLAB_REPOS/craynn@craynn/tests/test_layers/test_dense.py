def test_dense(tf):
  from craynn import layers

  input = layers.InputLayer(shape=(None, 32))
  dense1 = layers.dense(32)(input)
  dense2 = layers.dense(2)(dense1)

  def f(X):
    return layers.get_output(dense2, substitutes={ input : X })

  y = f(tf.random.normal(shape=(5, 32)))

  assert y.shape == (5, 2)
