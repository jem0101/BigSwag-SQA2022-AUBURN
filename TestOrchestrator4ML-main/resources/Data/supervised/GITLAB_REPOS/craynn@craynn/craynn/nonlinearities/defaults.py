from . import common

__all__ = [
  'default_semibounded_nonlinearity',
  'default_bounded_nonlinearity',
  'default_nonlinearity'
]

default_bounded_nonlinearity = common.sigmoid()

### well, leaky relu is not exactly semi-bounded...
default_semibounded_nonlinearity = common.leaky_relu(0.05)

default_nonlinearity = default_semibounded_nonlinearity