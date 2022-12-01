'''This package groups together a bunch of Theano code for neural nets.'''

from .activations import Activation
from .feedforward import Autoencoder, Regressor, Classifier
from .graph import Network
from .layers import Layer
from .losses import Loss
from .main import Experiment
from .regularizers import Regularizer
from .util import log

from . import convolution
from . import recurrent
from . import regularizers

__version__ = '0.8.0pre'
