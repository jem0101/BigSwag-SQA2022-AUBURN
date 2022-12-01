import tensorflow as tf

from .meta import Layer, model_from, get_output_shape

from ..parameters import zeros_init

__all__ = [
  'SoftmaxGate',
  'softmax_gate'
]

class SoftmaxGate(Layer):
  def __init__(self, *incoming, w=zeros_init(), name=None):
    incoming_shape = get_output_shape(incoming[0])
    self.w_broadcast = (None, ) * len(incoming_shape) + (Ellipsis, )

    super(SoftmaxGate, self).__init__(
      *incoming,
      name=name,
      parameters=(
        w(shape=(len(incoming),), weights=True, trainable=True, name='w')
      )
    )

  def get_output_for(self, w, *inputs):
    stacked = tf.stack(inputs, axis=-1)
    coefs = tf.nn.softmax(w)

    return tf.reduce_sum(stacked * coefs[self.w_broadcast], axis=-1)

  def get_output_shape_for(self, *input_shapes):
    return input_shapes[0]

softmax_gate = model_from(SoftmaxGate)()