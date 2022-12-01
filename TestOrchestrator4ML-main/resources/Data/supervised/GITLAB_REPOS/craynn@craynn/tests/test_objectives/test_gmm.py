from scipy import stats
import numpy as np

def test_gmm(tf):
  from craynn import objectives, variable_dataset
  from crayopt import tf_updates

  means = np.linspace(-2, 2, num=8)
  sigmas = np.exp(np.linspace(0, 1, num=8)) / 10

  X = np.concatenate([
    np.random.normal(size=(2048 // (i + 1), )) * sigmas[i] + means[i]
    for i in range(means.shape[0])
  ], axis=0).astype('float32')

  dataset = variable_dataset(X, )

  m = tf.Variable(
    initial_value=means.reshape(1, -1),
    dtype=tf.float32
  )

  s = tf.Variable(
    initial_value=np.log(sigmas).reshape(1, -1),
    dtype=tf.float32
  )

  p = tf.Variable(
    initial_value=np.zeros(8).reshape(1, -1),
    dtype=tf.float32
  )

  opt = tf_updates.adam(learning_rate=1e-2)([m, s, p])

  @tf.function
  def step():
    x, = dataset.batch(1)
    with opt:
      loss = objectives.gmm_neg_log_likelihood()(x, m, s, p)
      return opt(loss)

  for _ in range(32 * 1024):
    step()

  format = lambda xs: ', '.join(['%+.2lf' % x for x in xs])

  m = m.numpy()[0]
  s = s.numpy()[0]
  p = tf.nn.softmax(p).numpy()[0]

  indx0 = np.argsort(means)
  indx1 = np.argsort(m)

  print()
  print(format(means[indx0]))
  print(format(m[indx1]))
  print()
  print(format(sigmas[indx0]))
  print(format(np.exp(s[indx1])))
  print()
  print(format(p[indx1]))

  import matplotlib.pyplot as plt

  xs = np.linspace(np.min(X), np.max(X), num=1000)
  probs = np.sum([
    p[i] * stats.norm(loc=m[i], scale=np.exp(s[i])).pdf(xs.reshape(-1, 1))
    for i in range(8)
  ], axis=0)

  plt.figure()
  plt.hist(X, bins=100, histtype='step', density=True)
  plt.plot(xs, probs)
  plt.savefig('gmm.png')
  plt.close()