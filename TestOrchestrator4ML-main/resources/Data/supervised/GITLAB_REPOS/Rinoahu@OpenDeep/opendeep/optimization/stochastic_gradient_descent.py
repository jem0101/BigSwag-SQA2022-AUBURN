"""
Generic stochastic gradient descent optimization with momentum (Nesterov acceleration) and annealing.
"""
# standard libraries
import logging
# third party libraries
from theano.compat.python2x import OrderedDict  # use this compatibility OrderedDict
from six import iteritems
# internal references
from opendeep.utils.constructors import sharedX
from opendeep.optimization.optimizer import Optimizer
from opendeep.utils.decay import get_decay_function

log = logging.getLogger(__name__)


class SGD(Optimizer):
    """
    Stochastic gradient descent for training a model - includes learning rate decay and momentum.
    """
    def __init__(self, dataset, loss, model=None,
                 epochs=10, batch_size=100, min_batch_size=1,
                 save_freq=None, stop_threshold=None, stop_patience=None,
                 learning_rate=.1, lr_decay="exponential", lr_decay_factor=.995,
                 momentum=0.5, momentum_decay="linear", momentum_factor=0, nesterov_momentum=True,
                 grad_clip=None, hard_clip=False):
        """
        Initialize SGD.

        Parameters
        ----------
        dataset : Dataset
            The :class:`opendeep.data.Dataset` to use when training the Model.
        loss : Loss
            The :class:`opendeep.optimization.loss.Loss` function to compare the model to a 'target' result.
        model : Model
            The :class:`opendeep.models.Model` to train. Needed if the Optimizer isn't being passed to a
            Model's .train() method.
        epochs : int
            how many training iterations over the dataset to go.
        batch_size : int
            How many examples from the training dataset to use in parallel.
        min_batch_size : int
            The minimum number of examples required at a time (for things like time series, this would be > 1).
        save_freq : int
            How many epochs to train between each new save of the Model's parameters.
        stop_threshold : float
            The factor by how much the best validation training score needs to improve to determine early stopping.
        stop_patience : int
            The patience or number of epochs to wait after the stop_threshold has been reached before stopping.
        learning_rate : float
            The multiplicative amount to adjust parameters based on their gradient values.
        lr_decay : str
            The type of decay function to use for changing the learning rate over epochs. See
            `opendeep.utils.decay` for options.
        lr_decay_factor : float
            The amount to use for the decay function when changing the learning rate over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        momentum : float
            The momentum to use during gradient updates.
        momentum_decay : str
            The type of decay function to use for changing the momentum over epochs. See
            `opendeep.utils.decay` for options.
        momentum_factor : float
            The amount to use for the decay function when changing the momentum over epochs. See
            `opendeep.utils.decay` for its effect for given decay functions.
        nesterov_momentum : bool
            Whether or not to use Nesterov momentum.
        grad_clip : float, optional
            Whether to clip gradients. This will clip with a maximum of grad_clip or the parameter norm.
        hard_clip : bool
            Whether to use a hard cutoff or rescaling for clipping gradients.
        """
        # superclass init
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(SGD, self).__init__(**initial_parameters)

        # Momentum - smoothing over the parameter changes (see Hinton)
        if momentum:
            self.momentum = sharedX(momentum, 'momentum')
            if momentum_decay is not None and \
                            momentum_decay is not False and \
                            momentum_factor is not None:
                self.momentum_decay = get_decay_function(momentum_decay,
                                                         self.momentum,
                                                         self.momentum.get_value(),
                                                         momentum_factor)
            else:
                self.momentum_decay = False
        else:
            self.momentum = 0
            self.momentum_decay = False

        self.nesterov_momentum = nesterov_momentum

    def get_updates(self, gradients):
        """
        Based on Pylearn2
        (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py)

        Implements momentum as described in Section 9 of
        "A Practical Guide to Training Restricted Boltzmann Machines",
        Geoffrey Hinton.
        Parameters are updated by the formula:
        inc := momentum * inc - learning_rate * d cost / d param
        param := param + inc

        Also has the option to implement Nesterov momentum (accelerated momentum), which works better in a lot of cases.

        Parameters
        ----------
        gradients : dict
            A dictionary mapping from the model's parameters to their
            gradients.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.
        """
        log.debug('Setting up Stochastic Gradient Descent with momentum for optimizer...')
        updates = OrderedDict()
        for (param, gradient) in iteritems(gradients):
            velocity = sharedX(param.get_value() * 0.)

            assert param.dtype == velocity.dtype
            assert gradient.dtype == param.dtype

            if param.name is not None:
                velocity.name = 'vel_' + param.name

            scaled_lr = self.learning_rate * self.lr_scalers.get(param, 1.)
            updates[velocity] = self.momentum * velocity - scaled_lr * gradient

            inc = updates[velocity]
            if self.nesterov_momentum:
                log.debug('Using Nesterov momentum for parameter %s', str(param))
                inc = self.momentum * inc - scaled_lr * gradient

            assert inc.dtype == velocity.dtype
            updates[param] = param + inc

        return updates

    def get_decay_params(self):
        """
        Returns a list of all the Decay objects to decay during training.

        Returns
        -------
        list
            List of Decay objects to use after each training epoch - in this case the possibility to add
            momentum decay to the learning rate decay from the base optimizer class.
        """
        decay_params = super(SGD, self).get_decay_params()
        if hasattr(self, 'momentum_decay') and self.momentum_decay:
            decay_params.append(self.momentum_decay)
        return decay_params