"""
This module defines expressions for finding the loss or cost function for models to optimize.
"""
# standard libraries
import logging
# internal references
from opendeep.utils.misc import (raise_to_list, base_variables)

log = logging.getLogger(__name__)


class Loss(object):
    """
    The :class:`Loss` class takes a Theano expression and a target Theano symbolic variable to compute
    the loss function.

    Attributes
    ----------
    inputs : list
        List of theano symbolic expressions that are the necessary inputs to the loss function.
    targets : list
        List of target theano symbolic variables (or empty list) necessary for the loss function.
    args : dict
        Dictionary of all parameter arguments to the class initialization.
    """
    def __init__(self, inputs, targets=None, func=None, **kwargs):
        """
        Initializes the :class:`Loss` function.

        Parameters
        ----------
        inputs : list(theano symbolic expression)
            The input(s) necessary for the loss function.
        targets : list(theano symbolic variable), optional
            The target(s) variables for the loss function.
        func : function, optional
            A python function for computing the loss given the inputs list an targets list (in order).
            The function `func` will be called with parameters: func(*(list(inputs)+list(targets))).
        """
        self._classname = self.__class__.__name__
        log.debug("Creating a new instance of %s", self._classname)
        self.inputs = raise_to_list(inputs)
        if self.inputs is not None:
            ins = []
            # deal with Models or ModifyLayers being passed as an input.
            for input in self.inputs:
                if hasattr(input, 'get_outputs'):
                    inputs = raise_to_list(input.get_outputs())
                    for i in inputs:
                        ins.append(i)
                else:
                    ins.append(input)
            # replace self.inputs
            self.inputs = ins

        self.targets = raise_to_list(targets)
        if self.targets is None:
            self.targets = []
        self.func = func
        self.args = kwargs.copy()
        self.args['inputs'] = self.inputs
        self.args['targets'] = self.targets
        self.args['func'] = self.func

    def get_loss(self):
        """
        Returns the expression for the loss function.

        Returns
        -------
        theano expression
            The loss function.
        """
        if self.func is not None:
            inputs_targets = self.inputs + self.targets
            return self.func(*inputs_targets)
        else:
            raise NotImplementedError("Loss function not defined for %s" % self._classname)

    def get_targets(self):
        """
        Returns the target(s) Theano symbolic variables used to compute the loss. These will be fed
        into the training function and should have the right dtype for the intended data.

        Returns
        -------
        symbolic variable or list(theano symbolic variable)
            The symbolic variable(s) used when computing the loss (the target values). By default, this returns
            the `targets` parameter when initializing the :class:`Loss` class, raised to a list.
        """
        return self.targets

    def __add__(self, other):
        """
        Helper function to override adding behavior. Returns a new Loss trying to add the two get_loss() values,
        or this get_loss() value with the `other` added. Also modifies the inputs and targets based on `other`.
        """
        these_inputs = self.inputs or []
        if isinstance(other, Loss):
            def new_loss():
                return self.get_loss() + other.get_loss()

            def new_targets():
                return self.get_targets() + other.get_targets()

            those_inputs = other.inputs or []
            new_inputs = these_inputs + those_inputs

        else:
            def new_loss():
                return self.get_loss() + other

            def new_targets():
                return self.get_targets()

            new_inputs = these_inputs + list(base_variables(other))

        if len(new_inputs) is 0:
            new_inputs = None
        try:
            ret = Loss(inputs=new_inputs, targets=new_targets())
            ret.get_targets = new_targets
            ret.get_loss = new_loss
            return ret
        except Exception as e:
            log.exception("Exception when adding to the Loss class. {!s}".format(str(e)))
            raise

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """
        Helper function to override multiplication behavior. Returns a new Loss trying to multiply the two get_loss()
        values, or this get_loss() value with the `other` multiplied. Also modifies the inputs and targets
        based on `other`.
        """
        these_inputs = self.inputs or []
        if isinstance(other, Loss):
            def new_loss():
                return self.get_loss() * other.get_loss()

            def new_targets():
                return self.get_targets() + other.get_targets()

            those_inputs = other.inputs or []
            new_inputs = these_inputs + those_inputs
        else:
            def new_loss():
                return self.get_loss() * other

            def new_targets():
                return self.get_targets()

            new_inputs = these_inputs + list(base_variables(other))

        if len(new_inputs) is 0:
            new_inputs = None
        try:
            ret = Loss(inputs=new_inputs, targets=new_targets())
            ret.get_targets = new_targets
            ret.get_loss = new_loss
            return ret
        except Exception as e:
            log.exception("Exception when multiplying to the Loss class. {!s}".format(str(e)))
            raise

    def __rmul__(self, other):
        return self.__mul__(other)
