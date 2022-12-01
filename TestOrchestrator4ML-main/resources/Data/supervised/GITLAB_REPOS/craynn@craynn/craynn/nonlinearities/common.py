import tensorflow as tf

from .meta import nonlinearity_from

__all__ = [
  'sigmoid',
  'leaky_sigmoid',
  'tanh',
  'leaky_tanh',
  'relu',
  'leaky_relu',
  'softplus',
  'softmax',
  'elu',
  'linear',

  'gaussian'
]

sigmoid = nonlinearity_from(
  lambda name='sigmoid': lambda x: tf.nn.sigmoid(x, name=name)
)

leaky_sigmoid = nonlinearity_from(
  lambda leakiness=0.05, name='sigmoid': lambda x: \
    tf.nn.sigmoid(x, name=name) + leakiness * x
)

tanh = nonlinearity_from(
  lambda name='tanh': lambda x: tf.tanh(x, name=name)
)

leaky_tanh = nonlinearity_from(
  lambda leakiness=0.05, name='tanh': lambda x: \
    tf.tanh(x, name=name) + leakiness * x
)

relu = nonlinearity_from(
  lambda name='ReLU': lambda x: tf.nn.relu(x, name=name)
)

leaky_relu = nonlinearity_from(
  lambda leakiness=0.05, name='ReLU': \
    lambda x: tf.nn.leaky_relu(x, alpha=leakiness, name=name)
)

softplus = nonlinearity_from(
  lambda name='softplus': lambda x: tf.nn.softplus(x)
)
softmax = nonlinearity_from(
  lambda name='softmax': lambda x: tf.nn.softmax(x, name=name)
)

elu = nonlinearity_from(
  lambda name='ELU': lambda x: tf.nn.elu(x, name=name)
)

linear = nonlinearity_from(
  lambda name='linear': lambda x: x
)

gaussian = nonlinearity_from(
  lambda name='gaussian': lambda x: tf.exp(-x ** 2, name=name)
)