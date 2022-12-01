__all__ = [
  'glorot_dense'
]

from ..nonlinearities import linear
from ..parameters import glorot_normal_double_init, zeros_init

from .dense_ops import dense

glorot_dense = lambda num_units, target_shape, name=None: \
  dense(
    num_units,
    W=glorot_normal_double_init(gain=0.1, target_shape=target_shape),
    b=zeros_init(),
    activation=linear(),
    name=name
  )