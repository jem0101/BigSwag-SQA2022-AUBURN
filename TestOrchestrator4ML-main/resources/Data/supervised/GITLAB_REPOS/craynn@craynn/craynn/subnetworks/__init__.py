"""
General rules held in this module:
- params ending with `_op` like `conv_op` or simply op require function layer -> layer, e.g. conv(128, f=...)
- params not ending with `_op` like `conv` require function with similar signature as
    layer/subnetwork with the same name. Such parameters allow to override layer/subnetwork used.
    For example, overriding default nonlinearity: conv=lambda num_filters: conv(num_filters, f=T.nnet.sigmoid).
- usually, module provides two similar implementations of the same subnetwork like
  fire and fire_block. Subnetworks with shorter name are usually shortcuts to the method with longer name
  with popular default parameters.
"""

from .meta import *

from .skip import *
from .residual_nets import *
from .vae_nets import *

from .normalization import *

from .meta_nets import *
