from .common import normal_init, zeros_init
from .glorot import glorot_normal_init

__all__ = [
  'default_weight_init',
  'default_bias_init',
  'default_input_init'
]

default_input_init = normal_init()
default_weight_init = glorot_normal_init(gain=1.0)
default_bias_init = zeros_init()