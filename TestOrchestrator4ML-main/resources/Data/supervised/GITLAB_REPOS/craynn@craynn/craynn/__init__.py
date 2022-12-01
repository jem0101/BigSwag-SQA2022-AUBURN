"""
CRAYNN is just another layer of abstraction for neural networks.

Important note: this package is poorly designed, unstable and lacks documentation.
"""

from .nonlinearities.common import *
from .layers import *
from .parameters import *
from .updates import *
from .subnetworks import *
from .networks import *
from .regularization import *

from . import objectives

from . import utils

from . import viz

from . import info

try:
  import crayflow as datasets
except:
  class DelayedWarning(object):
    def __getattr__(self, item):
      import warnings
      warnings.warn('`craynn.datasets` is now a separate package. Please, visit https://gitlab.com/craynn/crayflow')

      raise ModuleNotFoundError('datasets')

  datasets = DelayedWarning()
