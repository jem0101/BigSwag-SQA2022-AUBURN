import tensorflow as tf

from ..nonlinearities import default_semibounded_nonlinearity
from ..parameters import default_weight_init, default_bias_init

from .meta import Layer, get_output_shape, model_from

__all__ = [
  'DenseLayer',
  'dense',

  'TensorDenseLayer',
  'tensor_dense',

  'BatchDenseLayer',
  'batch_dense',
]

class DenseLayer(Layer):
  def __init__(self, incoming, num_units,
               activation=default_semibounded_nonlinearity,
               W=default_weight_init,
               b=default_bias_init,
               name=None):
    """
    Dense (also called fully-connected) layer.
    Dense layer consists of `num_units` units, each of which takes
    weighted sum of inputs and applies `activation` function. In matrix form:
        f(X `dot` W + b)
    where:
      - W --- a weight matrix of size `(input_dim, num_units)`;
      - b --- a bias vector of size `(num_units, )`;
      - X --- input matrix of size `(batch_size, input_dim)`.

    If `X` has dimensionality `m > 2` first `m - 1` axes are treated as batch dimensions.

    :param incoming: incoming layer;
    :param num_units: number of output units;
    :param activation: activation function `f`;
    :param W: weight matrix, parameter with default properties `weights=True`, `trainable=True`;
    :param b: bias vector, parameter with default properties `biases=True`, `trainable=True`;
    :param name: name for the layer.
    """
    input_shape = get_output_shape(incoming)
    self.num_units = num_units

    if len(input_shape) < 2:
      raise ValueError('Dense layer accepts only tensors of dimensionality higher than 2 [got %s]!' % (input_shape, ))

    self.activation = activation

    super(DenseLayer, self).__init__(
      incoming,
      name=name,
      parameters=(
        W(shape=(input_shape[-1], num_units), name='W', weights=True, trainable=True),
        b(shape=(num_units,), name='b', biases=True, trainable=True)
      )
    )

  def get_output_for(self, W, b, input):
    b_broadcast = tuple(None for _ in input.shape[:-1]) + (slice(None, None, None), )

    return self.activation(
      tf.matmul(input, W) + b[b_broadcast]
    )

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    if len(input_shape) < 2:
      raise ValueError('Dense layer accepts only 2+ dimensional tensors!')

    return input_shape[:-1] + (self.num_units, )

dense = model_from(DenseLayer).with_fixed().with_defaults()()
meta_dense = model_from(DenseLayer, incoming_args=('incoming', 'W', 'b')).with_fixed().with_defaults()()

class TensorDenseLayer(Layer):
  def __init__(self, incoming, num_units,
               activation=default_semibounded_nonlinearity,
               W=default_weight_init,
               b=default_bias_init,
               axis=-1,
               name=None):
    input_shape = get_output_shape(incoming)
    self.num_units = num_units
    self.axis = (len(input_shape) + axis) % len(input_shape)

    self.b_broadcast = tuple(
      (None if i != self.axis else slice(None, None, None))
      for i in range(len(input_shape))
    )

    self.activation = activation

    super(TensorDenseLayer, self).__init__(
      incoming,
      name=name,
      parameters=(
        W(shape=(input_shape[self.axis], num_units), name='W', weights=True, trainable=True),
        b(shape=(num_units,), name='b', biases=True, trainable=True)
      ),
    )

  def get_output_for(self, W, b, input):
    return self.activation(
      tf.tensordot(input, W, axes=[(self.axis, ), (0, )]) + b[self.b_broadcast]
    )

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    return tuple([
      (input_shape[i] if i != self.axis else self.num_units)
      for i in range(len(input_shape))
    ])

tensor_dense = model_from(TensorDenseLayer).with_fixed().with_defaults()()


class BatchDenseLayer(Layer):
  def __init__(self, incoming, num_units, num_batches=None,
               activation=default_semibounded_nonlinearity,
               W=default_weight_init,
               b=default_bias_init,
               name=None):
    input_shape = get_output_shape(incoming)

    if num_batches is None:
      if len(input_shape) < 3:
        raise Exception('Please specify number of batches for the batch dense layer')

      self.num_batches = input_shape[0]
    else:
      self.num_batches = num_batches

    self.num_units = num_units
    self.in_units = input_shape[-1]

    self.activation = activation

    self.W = W(shape=(self.num_batches, self.in_units, num_units), name='W', weights=True, trainable=True)
    self.b = b(shape=(self.num_batches, num_units,), name='b', biases=True, trainable=True)

    super(BatchDenseLayer, self).__init__(
      incoming,
      name=name,
      parameters=(self.W, self.b)
    )

  def get_output_for(self, W, b, input):
    return self.activation(
      tf.matmul(input, W) + b[:, None, :]
    )

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    if len(input_shape) == 3:
      return (
        self.num_batches, input_shape[1], self.num_units
      )
    else:
      return (
        self.num_batches, input_shape[0], self.num_units
      )

batch_dense = model_from(BatchDenseLayer).with_fixed().with_defaults()()