def test_meta(tf):
  from craynn import layers, parameters, Network

  N = 7
  in_features = 11
  out_features = 13

  latent = 33

  X = layers.InputLayer(shape=(None, in_features), name='x')
  Z = layers.InputLayer(shape=(in_features, latent), name='z')

  p = layers.dense(out_features)(Z)
  meta_dense = layers.dense(out_features, W=parameters.bound_parameter(incoming=p))(X)

  net = Network(inputs=(X, Z), outputs=meta_dense)

  print(net(
    x=tf.random.normal(shape=(N, in_features)),
    z=tf.random.normal(shape=(in_features, latent)),
  ))