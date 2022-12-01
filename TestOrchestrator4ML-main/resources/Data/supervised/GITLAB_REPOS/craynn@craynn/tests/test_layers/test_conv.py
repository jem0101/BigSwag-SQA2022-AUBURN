def test_conv(gpu_tf):
  from craynn import layers

  tf = gpu_tf

  input = layers.InputLayer(shape=(None, 1, 32, 32))
  l1 = layers.conv(32)(input)
  l2 = layers.conv(2)(l1)

  @tf.function
  def f(X):
    return layers.get_output(l2, substitutes={ input : X })

  y = f(tf.random.normal(shape=(5, 1, 32, 32)))

  assert y.shape == (5, 2, 28, 28)

def test_conv_activation(gpu_tf):
  import inspect
  from craynn import layers

  tf = gpu_tf

  import numpy as np
  from craynn import network
  from craynn.nonlinearities import softplus

  print()
  print(inspect.signature(layers.Conv2DLayer))
  print(inspect.signature(layers.Conv2DLayer1x1))

  print(inspect.signature(layers.conv_2d))
  print(inspect.signature(layers.conv_2d_1x1))

  net = network((None, 3, 32, 32))(
    layers.conv_1x1(15, activation=softplus())
  )

  result = net(np.ones(shape=(2, 3, 32, 32), dtype='float32'))

  print(result)
  print(result.shape)