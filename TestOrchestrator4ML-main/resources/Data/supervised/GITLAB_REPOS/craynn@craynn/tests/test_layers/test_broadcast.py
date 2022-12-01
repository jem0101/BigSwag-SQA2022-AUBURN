import numpy as np

def test_broadcast_concat(tf):
  from craynn.layers import broadcast_concat, InputLayer, expand, get_output

  in1 = InputLayer((None, None, 3))
  in2 = InputLayer((None, 1))

  extend2 = expand[:, None, :](in2)
  concat = broadcast_concat()(in1, extend2)

  v1 = tf.random.normal(shape=(15, 7, 3))
  v2 = tf.random.normal(shape=(15, 1))

  result = get_output(concat, substitutes={
    in1 : v1,
    in2 : v2
  })

  print()
  print(tf.shape(result))

  assert np.allclose(
    result[:, :, :3].numpy(),
    v1.numpy()
  )

  for i in range(7):
    assert np.allclose(
      result[:, i, -1].numpy(),
      v2[:, 0].numpy()
    )