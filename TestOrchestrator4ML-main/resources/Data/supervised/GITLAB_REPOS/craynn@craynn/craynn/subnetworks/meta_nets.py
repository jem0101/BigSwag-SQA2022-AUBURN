from ..nonlinearities import default_nonlinearity
from ..parameters import unbound_parameter

from . import achain

__all__ = [
  'meta_dense',
  'meta_conv'
]

def meta_dense(num_units, meta_ops, activation=default_nonlinearity, name=None):
  from ..parameters import glorot_normal_init
  from ..layers import dense, glorot_dense, reshape, transpose

  def f(incoming):
    op = dense(
      num_units, activation=activation,
      W=unbound_parameter(),
      b=unbound_parameter()
    )(incoming)

    W, b = op.parameters
    W_shape, b_shape = W.shape, b.shape

    W_layer, b_layer = achain(
      meta_ops, [
        (glorot_dense(W_shape[0], target_shape=W_shape), transpose([1, 0])),
        (dense(1, W=glorot_normal_init(gain=0.1)), reshape(b_shape)),
      ]
    )()

    return MetaLayer((incoming, ), (W_layer, b_layer), op=op, name=name)

  return f

def meta_conv(
  num_filters, kernel_size=3, activation=default_nonlinearity,
  pad='valid', stride=1, dilation=1,
  meta_ops=None, name=None
):

  from ..parameters import glorot_normal_init
  from ..layers import conv, reshape, transpose, dense, glorot_dense

  def f(incoming):
    op = conv(
      num_filters, kernel_size=kernel_size,
      pad=pad, stride=stride, dilation=dilation,
      activation=activation,
      W=unbound_parameter(),
      b=unbound_parameter()
    )(incoming)

    W, b = op.parameters
    W_shape, b_shape = W.shape, b.shape
    W_shape_flat = W_shape[0] * W_shape[1] * W_shape[2]

    W_layer, b_layer = achain(
      meta_ops, [
        (glorot_dense(W_shape_flat, target_shape=W_shape), transpose([1, 0]), reshape(W_shape)),
        (dense(1, W=glorot_normal_init(gain=0.1)), reshape(b_shape)),
      ]
    )()

    return MetaLayer((incoming, ), (W_layer, b_layer), op=op, name=name)

  return f
