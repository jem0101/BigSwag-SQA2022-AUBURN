import tensorflow as tf

from ..parameters import glorot_scaling

__all__ = [
  'reg_l1', 'reg_l2', 'reg_elastic_net',
  'norm_preserving_l2'
]

def reg_l2():
  def reg(_, value):
    return tf.reduce_sum(value ** 2)

  return reg

def reg_l1():
  def reg(_, value):
    return tf.reduce_sum(tf.abs(value))

  return reg

def reg_elastic_net(alpha=0.1):
  l1 = reg_l1()
  l2 = reg_l2()

  def reg(param, value):
    return (1 - alpha) * l2(param, value) + alpha * l1(param, value)
  return reg


def norm_preserving_l2(scale_f=glorot_scaling):
  def reg(_, value):
    scale = scale_f(value.shape)
    norm = tf.sqrt(
      tf.reduce_mean(value ** 2)
    )

    return (norm - scale) ** 2

  return reg