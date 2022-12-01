"""
This module provides the most basic neural net layers. This goes from an input to an output with an optional activation.
"""
# standard libraries
import logging
# third party libraries
from theano.compat.python2x import OrderedDict
from theano.tensor import (dot, argmax, neq, mean)
import theano.sandbox.rng_mrg as RNG_MRG
# internal references
from opendeep.models.model import Model
from opendeep.utils.activation import get_activation_function
from opendeep.utils.decorators import inherit_docs
from opendeep.utils.weights import (get_weights, get_bias)

log = logging.getLogger(__name__)


@inherit_docs
class Dense(Model):
    """
    This is your basic input -> nonlinear(output) layer. No hidden representation. It is also known as a
    fully-connected layer.
    """
    def __init__(self, inputs=None, outputs=None, params=None, outdir='outputs/basic',
                 activation='rectifier',
                 weights_init='uniform', weights_mean=0, weights_std=5e-3, weights_interval='glorot',
                 bias_init=0.0,
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 **kwargs):
        """
        Initialize a basic layer.

        Parameters
        ----------
        inputs : List of [tuple(shape, `Theano.TensorType`)]
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. The length of `shape` should be equal to number of
            dimensions in `Theano.TensorType`, where the shape element is an integer representing the size for its
            dimension, or None if the shape isn't known. For example, if you have a matrix with unknown batch size
            but fixed feature size of 784, `shape` would be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)].
        outputs : int
            The dimensionality of the output for this model.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        activation : str or callable
            The activation function to use after the dot product going from input -> output. This can be a string
            representing an option from opendeep.utils.activation, or your own function as long as it is callable.
        weights_init : str
            Determines the method for initializing input -> output weights. See opendeep.utils.nnet for options.
        weights_interval : str or float
            If Uniform `weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        weights_mean : float
            If Gaussian `weights_init`, the mean value to use.
        weights_std : float
            If Gaussian `weights_init`, the standard deviation to use.
        bias_init : float
            The initial value to use for the bias parameter. Most often, the default of 0.0 is preferred.
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        """
        # init Model to combine the defaults and config dictionaries with the initial parameters.
        initial_parameters = locals().copy()
        initial_parameters.pop('self')
        super(Dense, self).__init__(**initial_parameters)
        if self.inputs is None:
            return

        ##################
        # specifications #
        ##################
        if len(self.inputs) > 1:
            raise NotImplementedError("Expected 1 input to Dense, found %d. Please merge inputs before passing "
                                      "to the Dense model!" % len(self.inputs))
        # self.inputs is a list of all the input expressions (we enforce only 1, so self.inputs[0] is the input)
        input_shape, self.input = self.inputs[0]
        if isinstance(input_shape, int):
            self.input_size = ((None, ) * (self.input.ndim-1)) + (input_shape, )
        else:
            self.input_size = input_shape
        assert self.input_size is not None, "Need to specify the shape for the last dimension of the input!"

        # We also only have 1 output
        assert self.output_size is not None, "Need to specify outputs size!"
        out_size = self.output_size[0]
        if isinstance(out_size, int):
            self.output_size = self.input_size[:-1] + (out_size,)
        else:
            self.output_size = out_size

        # activation function!
        activation_func = get_activation_function(activation)

        #########################################################
        # parameters - make sure to deal with input dictionary! #
        #########################################################
        W = self.params.get("W") or get_weights(
            weights_init=weights_init,
            shape=(self.input_size[-1], self.output_size[-1]),
            name="W",
            rng=mrg,
            # if gaussian
            mean=weights_mean,
            std=weights_std,
            # if uniform
            interval=weights_interval
        )

        b = self.params.get("b") or get_bias(shape=self.output_size[-1], name="b", init_values=bias_init)


        # Finally have the two parameters - weights matrix W and bias vector b. That is all!
        self.params = OrderedDict([("W", W), ("b", b)])

        ###############
        # computation #
        ###############
        # Here is the meat of the computation transforming input -> output
        # It simply involves a matrix multiplication of inputs*weights, adding the bias vector, and then passing
        # the result through our activation function (normally something nonlinear such as: max(0, output))
        self.output = activation_func(dot(self.input, W) + b)

        log.debug("Initialized a basic fully-connected layer with shape %s and activation: %s",
                  str((self.input_size[-1], self.output_size[-1])), str(activation))

    def get_inputs(self):
        return self.input

    def get_outputs(self):
        return self.output

    def get_params(self):
        return self.params


@inherit_docs
class Softmax(Dense):
    """
    The softmax layer is meant as a last-step prediction layer using the softmax activation function -
    this class exists to provide easy access to methods for errors and log-likelihood for a given truth label y.

    It is a special subclass of the Dense (a fully-connected layer),
    with the activation function forced to be 'softmax'

    Attributes
    ----------
    p_y_given_x : theano expression
        Theano expression for the probabilities of the class labels
    y_pred : theano expression
        Theano expression for the predicted class number (argmax of p_y_given_x)
    """
    def __init__(self, inputs=None, outputs=None, params=None, outdir='outputs/softmax',
                 weights_init='uniform', weights_mean=0, weights_std=5e-3, weights_interval='glorot',
                 bias_init=0.0,
                 out_as_probs=True,
                 mrg=RNG_MRG.MRG_RandomStreams(1),
                 **kwargs):
        """
        Initialize a Softmax layer.

        Parameters
        ----------
        inputs : List of [tuple(shape, `Theano.TensorType`)]
            The dimensionality of the inputs for this model, and the routing information for the model
            to accept inputs from elsewhere. `shape` will be a monad tuple representing known
            sizes for each dimension in the `Theano.TensorType`. The length of `shape` should be equal to number of
            dimensions in `Theano.TensorType`, where the shape element is an integer representing the size for its
            dimension, or None if the shape isn't known. For example, if you have a matrix with unknown batch size
            but fixed feature size of 784, `shape` would be: (None, 784). The full form of `inputs` would be:
            [((None, 784), <TensorType(float32, matrix)>)].
        outputs : int
            The dimensionality of the output for this model.
        params : Dict(string_name: theano SharedVariable), optional
            A dictionary of model parameters (shared theano variables) that you should use when constructing
            this model (instead of initializing your own shared variables). This parameter is useful when you want to
            have two versions of the model that use the same parameters - such as siamese networks or pretraining some
            weights.
        outdir : str
            The directory you want outputs (parameters, images, etc.) to save to. If None, nothing will
            be saved.
        weights_init : str
            Determines the method for initializing input -> output weights. See opendeep.utils.nnet for options.
        weights_interval : str or float
            If Uniform `weights_init`, the +- interval to use. See opendeep.utils.nnet for options.
        weights_mean : float
            If Gaussian `weights_init`, the mean value to use.
        weights_std : float
            If Gaussian `weights_init`, the standard deviation to use.
        bias_init : float
            The initial value to use for the bias parameter. Most often, the default of 0.0 is preferred.
        out_as_probs : bool
            Whether to output the argmax prediction (the predicted class of the model), or the probability distribution
            over all classes. True means output the distribution of size `output_size` and False means output a single
            number index for the class that had the highest probability.
        mrg : random
            A random number generator that is used when adding noise.
            I recommend using Theano's sandbox.rng_mrg.MRG_RandomStreams.
        """
        # init the fully connected generic layer with a softmax activation function
        super(Softmax, self).__init__(inputs=inputs,
                                      outputs=outputs,
                                      params=params,
                                      outdir=outdir,
                                      activation='softmax',
                                      weights_init=weights_init,
                                      weights_mean=weights_mean,
                                      weights_std=weights_std,
                                      weights_interval=weights_interval,
                                      bias_init=bias_init,
                                      out_as_probs=out_as_probs,
                                      mrg=mrg,
                                      **kwargs)
        if self.inputs is None:
            return
        # the outputs of the layer are the probabilities of being in a given class
        self.p_y_given_x = super(Softmax, self).get_outputs()
        self.y_pred = argmax(self.p_y_given_x, axis=1)

        if out_as_probs:
            self.output = self.p_y_given_x
        else:
            self.output = self.y_pred

        self.out_as_probs = out_as_probs

    def errors(self):
        """
        Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero-one
        loss over the size of the minibatch.

        Returns
        -------
        float
            The error amount.
        """
        if self.out_as_probs:
            return mean(neq(self.y_pred, argmax(self.target, axis=1)))
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return mean(neq(self.y_pred, self.target))

    def get_argmax_prediction(self):
        """
        Returns the index of the class with the highest probability output.

        Returns
        -------
        int
            Index of the class with the highest probability.
        """
        # return the argmax y_pred class
        return self.y_pred
