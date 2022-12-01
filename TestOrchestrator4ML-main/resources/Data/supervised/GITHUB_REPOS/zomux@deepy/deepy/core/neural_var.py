#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano.tensor as TT

from deepy.core.tensor_conversion import neural_computation, convert_to_theano_var
from deepy.layers.layer import NeuralLayer


class NeuralVariable(NeuralLayer):
    """
    Create a constant layer with tensors.
    """

    def __init__(self, tensor, dim=0):
        """
        Create a tensor layer.
        :type tensor: theano.tensor.var.TensorVariable
        """
        super(NeuralVariable, self).__init__("const")
        self.output_dim = dim
        self.tensor = tensor
        self.init(0)


    def __getattr__(self, name):
        return NeuralVariable(getattr(self.tensor, name), dim=self.dim())

    def apply(self, func, dim=None):
        """
        Apply a function to tensors.
        """
        output_dim = dim if dim else self.output_dim
        return NeuralVariable(func(self.tensor), output_dim)

    def compute_tensor(self, x):
        return self.tensor

    def set_test_value(self, value):
        self.tensor.tag.test_value = value

    def dim(self):
        return self.output_dim

    def _other_tensor(self, other):
        return  other.tensor if isinstance(other, NeuralVariable) else other

    def __eq__(self, other):
        return NeuralVariable(TT.eq(self.tensor, self._other_tensor(other)), dim=self.dim())

    def __ne__(self, other):
        return NeuralVariable(TT.neq(self.tensor, self._other_tensor(other)), dim=self.dim())

    def __add__(self, other):
        return NeuralVariable(self.tensor + self._other_tensor(other), dim=self.dim())

    def __sub__(self, other):
        return NeuralVariable(self.tensor - self._other_tensor(other), dim=self.dim())

    def __mul__(self, other):
        return NeuralVariable(self.tensor * self._other_tensor(other), dim=self.dim())

    def __div__(self, other):
        return NeuralVariable(self.tensor / self._other_tensor(other), dim=self.dim())

    def __neg__(self):
        return NeuralVariable(- self.tensor, dim=self.dim())

    def __radd__(self, other):
        return NeuralVariable(self._other_tensor(other) + self.tensor, dim=self.dim())

    def __rsub__(self, other):
        return NeuralVariable(self._other_tensor(other) - self.tensor, dim=self.dim())

    def __rmul__(self, other):
        return NeuralVariable(self._other_tensor(other) * self.tensor, dim=self.dim())

    def __rdiv__(self, other):
        return NeuralVariable(self._other_tensor(other) / self.tensor, dim=self.dim())

    def __getitem__(self, index):
        @neural_computation
        def getitem_wrapper(t, index):
            if type(index) == list:
                index = tuple(index)
            return t.__getitem__(index)
        ret = getitem_wrapper(self, index)
        if (hasattr(ret.tensor, 'tag') and hasattr(ret.tensor.tag, 'test_value')
            and ret.tensor.tag.test_value is not None and len(ret.tensor.tag.test_value.shape) > 0):
            ret.output_dim = ret.tensor.tag.test_value.shape[-1]
        else:
            ret.output_dim = self.dim()
        return ret

    def __call__(self, *args, **kwargs):
        normal_args, tensor_found_in_args, neural_found_in_args = convert_to_theano_var(args)
        normal_kwargs, tensor_found_in_kwargs, neural_found_in_kwargs = convert_to_theano_var(kwargs)

        tensor_found = tensor_found_in_args or tensor_found_in_kwargs

        if tensor_found:
            raise Exception("Theano tensor variables can not be used together with neural variables.")

        return NeuralVariable(self.tensor(*normal_args, **normal_kwargs), dim=self.dim())

    def debug_monitor(self, name=""):
        from deepy.debug import monitor_var_sum
        self.tensor += monitor_var_sum(self.tensor, name=name)

    @property
    def test_value(self):
        if hasattr(self.tensor.tag, 'test_value'):
            return self.tensor.tag.test_value
        else:
            return None

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def tv(self):
        return self.test_value

    @property
    def ts(self):
        if self.test_value is not None:
            return self.test_value.shape
        else:
            return None