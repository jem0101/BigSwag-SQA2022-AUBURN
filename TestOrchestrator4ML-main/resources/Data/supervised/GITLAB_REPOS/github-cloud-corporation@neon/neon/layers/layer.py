# ----------------------------------------------------------------------------
# Copyright 2014-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
from __future__ import division
from builtins import str, zip
import logging
import numpy as np

from neon import NervanaObject
from neon.backends import Autodiff
from neon.backends.backend import Tensor
from neon.util.persist import load_class


logger = logging.getLogger(__name__)


def interpret_in_shape(xshape):
    """
    Helper function to interpret the tensor layout of preceding layer to handle non-recurrent,
    recurrent, and local layers.
    """
    if isinstance(xshape, (int, np.integer)):
        return (xshape, 1)
    else:
        if len(xshape) == 2:
            return xshape
        else:
            return (np.prod(xshape), 1)


class Layer(NervanaObject):

    """
    Top level generic neural network layer class from which all other layer
    types inherit.

    Arguments:
        name (string): Name identifying this layer (in logs, etc.)
        parallelism (int): Type of parallelism preferred by this layer. Possible
            values are "Unknown", "Disabled", and "Data". Only applicable to
            distributed backends (see gen_backend for details).
    """

    def __init__(self, name=None, parallelism="Unknown"):
        super(Layer, self).__init__(name)
        self.outputs = None
        self.has_params = False
        self.inputs = None
        self.owns_output = True
        self.owns_delta = False
        self.deltas = None
        self.parallelism = parallelism
        self.revert_list = []
        self.next_layer = None
        self.actual_bsz = None
        self.actual_seq_len = None

    def __str__(self):
        """
        Format the layer as a printable string.
        """
        ret = '{} {}'.format(self.classnm, self.name)
        return ret

    def nested_str(self, level=0):
        """
        Utility function for displaying layer info with a given indentation level.

        Arguments:
            level (int, optional): indentation level

        Returns:
            str: layer info at the given indentation level
        """

        return "  " * level + str(self)

    def configure(self, in_obj):
        """
        Set shape based parameters of this layer given an input tuple, int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer, Tensor or dataset): object that provides shape
                                                           information for layer
        """
        if isinstance(in_obj, Layer):
            self.prev_layer = in_obj
            self.in_shape = in_obj.out_shape
            if self.parallelism == "Unknown":
                self.parallelism = in_obj.parallelism
        else:
            self.prev_layer = None
            if isinstance(in_obj, (tuple, int, list)):
                self.in_shape = in_obj  # input is a shape tuple or int directly
            elif isinstance(in_obj, Tensor):
                self.in_shape = (in_obj.shape[0], in_obj.shape[1] // self.be.bsz)
            else:
                self.in_shape = in_obj.shape  # This is a dataset

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.
        Don't reallocate if it already exists.
        Only allocate space if layer owns its own output (e.g., bias and activation work
        in place, so do not own their output).
        Outputs can be allocated from a pre-allocated pool if shared_outputs is provided.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into

        """
        if self.outputs:
            return
        if self.owns_output:
            self.outputs = self.be.iobuf(self.out_shape, shared=shared_outputs,
                                         parallelism=self.parallelism)

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        if self.next_layer is not None and self.next_layer.parallelism != self.parallelism:
            self.owns_delta = True

        if self.owns_delta and self.prev_layer:
            if type(self.prev_layer) in (BranchNode,):
                self.deltas = self.prev_layer.deltas
            else:
                self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers[0],
                                            parallelism=self.parallelism)
                delta_buffers.reverse()
        else:
            self.deltas = None

    def set_next(self, layer):
        """
        Set next_layer to provided layer.

        Arguments:
            layer (layer): Next layer
        """
        self.next_layer = layer

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def _fprop_inference(self, inputs):
        """
        Apply the forward pass transformation to the input data.

        May skip any computation not needed for doing inference only.

        Calling bprop subsequently is not valid.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        raise NotImplementedError

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        raise NotImplementedError

    def get_terminal(self):
        """
        Used for recursively getting final nodes from layer containers.
        """
        return self

    def serialize(self):
        """
        Get state parameters for this layer.

        Returns:
            varies: whatever data this model wants to receive in order to restore state
        """
        if self.has_params:
            return self.get_params()

    def load_weights(self, pdict, load_states=True):
        """
        Load weights.

        Arguments:
            pdict:
            load_states:  (Default value = True)
        """
        self.set_params(pdict)
        if load_states:
            self.set_states(pdict)

    def get_param_attrs(self):
        return dict(parallel=(self.parallelism in ("Data", "Model")),
                    distributed=(self.parallelism == "Model"))

    def set_params(self, pdict):
        pass

    def set_states(self, pdict):
        pass

    def set_batch_size(self, N):
        """
        Set minibatch size.

        Arguments:
            N (int): minibatch size
        """
        self.actual_bsz = N

    def set_seq_len(self, S):
        """
        Set sequence length.

        Arguments:
            S (int): sequence length
        """
        self.actual_seq_len = S

    def get_description(self, **kwargs):
        return super(Layer, self).get_description(**kwargs)


class BranchNode(Layer):
    """
    Layer that allows branching.  Used to send outputs to multiple layer pathways.
    Each pathway will get the entire output of the layer preceding the branch node.
    """
    instances = {}

    def __new__(cls, name=None):
        """
        Branch nodes need to have a unique name,
        which identifies them.  This method checks
        to see if the branch node being made has already
        been created using its name and if so it returns
        that instance.
        """
        if name in cls.instances:
            return cls.instances[name]
        else:
            return Layer.__new__(cls)

    def __init__(self, name=None):
        # don't init if this is not a new instance
        # see __new__ above
        if name not in BranchNode.instances:
            BranchNode.instances[name] = self
            super(BranchNode, self).__init__(name)
            self.owns_output = False

    def fprop(self, inputs=None, inference=False):
        """
        Passes output from preceding layer on without modification.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        if self.outputs is None and inputs is not None:
            self.outputs = inputs
        return self.outputs

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        if hasattr(self, 'in_shape') and in_obj is None:
            return self  # previously configured, so just return
        super(BranchNode, self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        if self.deltas is None:
            self.deltas = self.be.iobuf(self.in_shape, shared=delta_buffers[0])
            delta_buffers.reverse()

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Branch nodes should be skipped in bprop, since their deltas are shared

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        pass


class SkipNode(Layer):
    """
    Layer that allows pass-through as in [He2015]_.

    Notes:

    .. [He2015] http://arxiv.org/abs/1502.03167
    """

    def __init__(self, name=None):
        super(SkipNode, self).__init__(name)
        self.owns_delta = True

    def fprop(self, inputs=None, inference=False, beta=0):
        """
        Passes output from preceding layer on without modification

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): Scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.outputs[:] = self.outputs * beta + inputs
        return self.outputs

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(SkipNode, self).configure(in_obj)
        self.out_shape = self.in_shape
        return self

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Skip nodes just pass back what they got.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.deltas[:] = self.deltas * beta + alpha * error
        return self.deltas


class Pooling(Layer):

    """
    Pooling layer implementation.

    Arguments:
        fshape (int, tuple(int, int)): one or two dimensional shape
            of pooling window
        op (str, optional): pooling operation in [max, avg]. Defaults to "max"
        strides (int, dict, optional): strides to apply pooling window
            over. An int applies to both dimensions, or a dict with str_h
            and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        name (str, optional): layer name. Defaults to "PoolingLayer"
    """

    def __init__(self, fshape, op="max", strides={}, padding={},
                 name=None):
        super(Pooling, self).__init__(name)
        self.poolparams = {'str_h': None, 'str_w': None, 'str_d': None, 'str_c': None,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0, 'pad_c': 0,
                           'J': 1, 'T': 1, 'D': 1, 'op': op}  # 3D paramaters

        # keep args around in __dict__ for get_description
        self.op = op
        self.fshape = fshape
        self.strides = strides
        self.padding = padding
        self.owns_delta = True
        if isinstance(fshape, int):
            fshape = {'R': fshape, 'S': fshape}
        elif isinstance(fshape, tuple):
            fkeys = ('R', 'S') if len(fshape) == 2 else ('T', 'R', 'S')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        elif fshape == 'all':
            fshape = dict(R=None, S=None)
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.poolparams.update(d)
        self.nglayer = None

    def __str__(self):
        return "Pooling Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs" % (
               self.name,
               self.in_shape[0], self.in_shape[1], self.in_shape[2],
               self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Pooling, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.poolparams.update(shapedict)
            if self.poolparams['R'] is None:
                self.poolparams['R'] = shapedict['H']
                self.poolparams['S'] = shapedict['W']
            self.nglayer = self.be.pool_layer(self.be.default_dtype, **self.poolparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, M, P, Q) if len(self.in_shape) == 4 else (K, P, Q)
        return self

    def set_deltas(self, delta_buffers):
        """
        Use pre-allocated (by layer containers) list of buffers for backpropagated error.
        Only set deltas for layers that own their own deltas
        Only allocate space if layer owns its own deltas (e.g., bias and activation work in-place,
        so do not own their deltas).

        Arguments:
            delta_buffers (list): list of pre-allocated tensors (provided by layer container)
        """
        super(Pooling, self).set_deltas(delta_buffers)
        if self.op == "max":
            self.argmax = self.be.empty(self.outputs.shape, dtype=np.uint8)
        else:
            self.argmax = None

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_pool(self.nglayer, inputs, self.outputs, self.argmax, beta=beta)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.be.bprop_pool(self.nglayer, error, self.deltas, self.argmax, alpha, beta)
        return self.deltas


class ParameterLayer(Layer):

    """
    Intermediate class used for common functionality for any layer with weights.

    Not intended to be used directly.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ParameterLayer"
    """

    def __init__(self, init=None, name=None,
                 parallelism="Unknown"):
        super(ParameterLayer, self).__init__(name, parallelism)
        self.has_params = True
        self.init = init
        self.W = None
        self.dW = None
        self.weight_shape = None
        self.batch_sum = None
        self.batch_sum_shape = None
        self.states = []
        self.owns_delta = True

    @classmethod
    def gen_class(cls, pdict):
        if 'init' in pdict and pdict['init'] is not None:
            cname = pdict['init']['type']
            icls = load_class(cname)
            init = icls(**pdict['init']['config'])
            pdict['init'] = init
        return cls(**pdict)

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(ParameterLayer, self).allocate(shared_outputs)
        if self.W is None:
            self.init_params(self.weight_shape)
        if self.batch_sum_shape is not None:
            self.batch_sum = self.be.empty(self.batch_sum_shape, dtype=np.float32,
                                           **self.get_param_attrs())

    def init_params(self, shape):
        """
        Allocate layer parameter buffers and initialize them with the
            supplied initializer.

        Arguments:
            shape (int, tuple): shape to allocate for layer parameter
                buffers.
        """
        self.W = self.be.empty(shape, **self.get_param_attrs())
        self.dW = self.be.empty_like(self.W)
        self.states = []

        if isinstance(self.init, Tensor) or isinstance(self.init, np.ndarray):
            assert self.init.shape == self.W.shape, "Initial weights shape does not match"
            self.W[:] = self.init
        else:
            self.init.fill(self.W)

    def get_params(self):
        """
        Get layer parameters, gradients, and states for optimization.
        """
        return ((self.W, self.dW), self.states)

    def get_params_serialize(self, keep_states=True):
        return self.get_description(get_weights=True, keep_states=keep_states)

    def get_description(self, get_weights=False, keep_states=True):
        """
        Get layer parameters. All parameters are needed for optimization, but
        only weights are serialized.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Control whether all parameters are returned
                or just weights for serialization.
        """
        serial_dict = super(ParameterLayer, self).get_description()
        if get_weights:
            serial_dict['params'] = {'W': self.W.get()}
            if keep_states:
                serial_dict['states'] = [s.get() for s in self.states]
        return serial_dict

    def set_params(self, pdict):
        """
        Set layer parameters (weights). Allocate space for other parameters but do not initialize
        them.

        Arguments:
            pdict (dict, ndarray): dictionary or ndarray with layer parameters
                                   [support for ndarray is DEPRECATED and will be removed]
        """
        assert type(pdict) is dict
        for key in pdict['params']:
            if not hasattr(self, key):
                setattr(self, key, None)

            attr = getattr(self, key)
            if isinstance(attr, Tensor):
                # this attr has already been allocated
                # get set the values
                attr.set(pdict['params'][key])
            elif type(pdict['params'][key]) is np.ndarray:
                setattr(self, key, self.be.array(pdict['params'][key], **self.get_param_attrs()))
            else:
                setattr(self, key, pdict['params'][key])

        if self.dW is None:
            self.dW = self.be.empty_like(self.W)

    def set_states(self, pdict):
        if 'states' not in pdict:
            # if states was not serialized then leave
            # this empty, the optimizer will initialize it
            self.states = []
        else:
            # this needs to be done in two steps for MGPU backend
            if self.states is None or len(self.states) == 0:
                self.states = [self.be.zeros_like(self.dW)
                               for i in range(len(pdict['states']))]

            for ind in range(len(pdict['states'])):
                self.states[ind].set(pdict['states'][ind])


class Convolution(ParameterLayer):

    """
    Convolutional layer implementation.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "ConvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name=None, parallelism="Data"):
        super(Convolution, self).__init__(init, name, parallelism)
        self.nglayer = None
        self.bsum = bsum
        self.convparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                           'pad_h': 0, 'pad_w': 0, 'pad_d': 0,
                           'T': 1, 'D': 1}  # 3D paramaters

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple) or isinstance(fshape, list):
            fkeys = ('R', 'S', 'K') if len(fshape) == 3 else ('T', 'R', 'S', 'K')
            fshape = {k: x for k, x in zip(fkeys, fshape)}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.convparams.update(d)

    def __str__(self):
        spatial_dim = len(self.in_shape[1:])
        spatial_str = "%d x (" + "x".join(("%d",) * spatial_dim) + ")"
        padstr_str = ",".join(("%d",) * spatial_dim)
        padstr_dim = ([] if spatial_dim == 2 else ['d']) + ['h', 'w']

        pad_tuple = tuple(self.convparams[k] for k in ['pad_' + d for d in padstr_dim])
        str_tuple = tuple(self.convparams[k] for k in ['str_' + d for d in padstr_dim])

        fmt_tuple = (self.name,) + self.in_shape + self.out_shape + pad_tuple + str_tuple
        fmt_string = "Convolution Layer '%s': " + \
                     spatial_str + " inputs, " + spatial_str + " outputs, " + \
                     padstr_str + " padding, " + padstr_str + " stride"

        return ((fmt_string % fmt_tuple))

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Convolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.convparams.update(shapedict)
            self.nglayer = self.be.conv_layer(self.be.default_dtype, **self.convparams)
            (K, M, P, Q, N) = self.nglayer.dimO
            self.out_shape = (K, P, Q) if M == 1 else (K, M, P, Q)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.bsum:
            self.batch_sum_shape = (self.nglayer.K, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_conv(self.nglayer, inputs, self.W, self.outputs, beta=beta,
                           bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_conv(self.nglayer, self.W, error, self.deltas,
                               alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, self.inputs, error, self.dW)
        return self.deltas


class Deconvolution(ParameterLayer):

    """
    Deconvolutional layer implementation.

    Arguments:
        fshape (tuple): three dimensional shape of convolution window
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        padding (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): layer name. Defaults to "DeconvolutionLayer"
    """

    def __init__(self, fshape, strides={}, padding={}, init=None, bsum=False,
                 name=None):
        super(Deconvolution, self).__init__(init, name)
        self.nglayer = None
        self.bsum = bsum
        self.deconvparams = {'str_h': 1, 'str_w': 1, 'str_d': 1,
                             'pad_h': 0, 'pad_w': 0, 'pad_d': 0}

        # keep around args in __dict__ for get_description.
        self.fshape = fshape
        self.strides = strides
        self.padding = padding

        if isinstance(fshape, tuple):
            # fshape[2] should now map to C (nifm)
            fshape = {'R': fshape[0], 'S': fshape[1], 'C': fshape[2]}
        if isinstance(strides, int):
            strides = {'str_h': strides, 'str_w': strides}
        if isinstance(padding, int):
            padding = {'pad_h': padding, 'pad_w': padding}
        for d in [fshape, strides, padding]:
            self.deconvparams.update(d)

    def __str__(self):
        return "Deconvolution Layer '%s': %d x (%dx%d) inputs, %d x (%dx%d) outputs" % (
               self.name,
               self.in_shape[0], self.in_shape[1], self.in_shape[2],
               self.out_shape[0], self.out_shape[1], self.out_shape[2])

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Deconvolution, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            shapedict = {'K': self.in_shape[0],
                         'P': self.in_shape[1],
                         'Q': self.in_shape[2],
                         'N': self.be.bsz}
            self.deconvparams.update(shapedict)
            self.nglayer = self.be.deconv_layer(self.be.default_dtype, **self.deconvparams)
            self.out_shape = (self.nglayer.C, self.nglayer.H, self.nglayer.W)
        if self.weight_shape is None:
            self.weight_shape = self.nglayer.dimF2  # (C * R * S, K)
        if self.bsum:
            self.batch_sum_shape = (self.nglayer.C, 1)
        return self

    def fprop(self, inputs, inference=False):
        """
        fprop for deconv is equivalent to bprop for conv.
        bprop_conv takes in error and deltas as "E" and "grad_I"
        for deconv, bprop_conv will take in input as "E" and output as "grad_I"

        Arguments:
            inference (bool): is inference only
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.bprop_conv(layer=self.nglayer, F=self.W, E=inputs, grad_I=self.outputs,
                           bsum=self.batch_sum)
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        bprop for deconv is equivalent to fprop for conv.
        fprop_conv takes input and output as "I" and "O".
        for deconv, fprop_conv will take error as input and delta as output

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.fprop_conv(self.nglayer, error, self.W, self.deltas, alpha=alpha, beta=beta)
        self.be.update_conv(self.nglayer, error, self.inputs, self.dW)
        return self.deltas


class Linear(ParameterLayer):

    """
    A fully connected layer implemented as the dot product of inputs and
    weights.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): Layer name. Defaults to "LinearLayer"
    """

    def __init__(self, nout, init, bsum=False, name=None, parallelism="Disabled"):
        super(Linear, self).__init__(init, name, parallelism)
        self.nout = nout
        self.inputs = None
        self.bsum = bsum

    def __str__(self):
        return "Linear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Linear, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.nout, self.nsteps)
        if self.weight_shape is None:
            self.weight_shape = (self.nout, self.nin)
        if self.bsum:
            self.batch_sum_shape = (self.nout, 1)
        return self

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only
            beta (float, optional): scale to apply to the outputs

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        if self.actual_bsz is None and self.actual_seq_len is None:
            self.be.compound_dot(A=self.W, B=self.inputs, C=self.outputs, beta=beta,
                                 bsum=self.batch_sum)
        else:
            bsz = self.be.bsz if self.actual_bsz is None else self.actual_bsz
            steps = self.nsteps if self.actual_seq_len is None else self.actual_seq_len

            self.be.compound_dot(A=self.W,
                                 B=self.inputs[:, :bsz * steps],
                                 C=self.outputs[:, :bsz * steps],
                                 beta=beta,
                                 bsum=self.batch_sum)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.compound_dot(A=self.W.T, B=error, C=self.deltas, alpha=alpha, beta=beta)
        self.be.compound_dot(A=error, B=self.inputs.T, C=self.dW)
        return self.deltas


class BinaryLinear(Linear):

    """
    A binary fully connected layer implemented as the dot product of inputs
    and binarized weights.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights
        name (str, optional): Layer name. Defaults to "BinaryLinearLayer"
    """

    def __str__(self):
        return "BinaryLinear Layer '%s': %d inputs, %d outputs" % (
               self.name, self.nin, self.nout)

    def allocate(self, shared_outputs=None):
        super(BinaryLinear, self).allocate(shared_outputs)
        self.Wb = self.be.empty_like(self.W)

    def fprop(self, inputs, inference=False, beta=0.0):
        self.inputs = inputs
        self.be.binarize(self.W, self.Wb, stochastic=False)

        not_binarized = self.be.zeros(self.inputs.shape)
        not_binarized[:] = self.be.not_equal(self.be.absolute(self.inputs), 1)
        if np.any(not_binarized.get()):
            gemm = self.be.compound_dot
        else:
            gemm = self.be.xnor_compound_dot

        if self.actual_bsz is None and self.actual_seq_len is None:
            gemm(A=self.Wb, B=self.inputs, C=self.outputs, beta=beta,
                 bsum=self.batch_sum)
        else:
            bsz = self.be.bsz if self.actual_bsz is None else self.actual_bsz
            steps = self.nsteps if self.actual_seq_len is None else self.actual_seq_len

            gemm(A=self.Wb,
                 B=self.inputs[:, :bsz * steps],
                 C=self.outputs[:, :bsz * steps],
                 beta=beta,
                 bsum=self.batch_sum)

        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        if self.deltas:
            self.be.compound_dot(A=self.Wb.T, B=error, C=self.deltas, alpha=alpha, beta=beta)
        self.be.compound_dot(A=error, B=self.inputs.T, C=self.dW)
        return self.deltas


class Bias(ParameterLayer):

    """
    A bias layer implemented that adds a learned bias to inputs and produces
    outputs of the same shape.

    Arguments:
        init (Initializer, optional): Initializer object to use for
            initializing layer bias
        name (str, optional): Layer name. Defaults to "BiasLayer"
    """

    def __init__(self, init, name=None):
        super(Bias, self).__init__(init, name)
        self.y = None
        self.owns_output = False
        self.owns_delta = False

    def __str__(self):
        if len(self.in_shape) == 3:
            layer_string = "Bias Layer '%s': size %d x (%dx%d)" % (
                self.name, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        else:
            layer_string = "Bias Layer '%s': size %d" % (self.name, self.bias_size)
        return layer_string

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Bias, self).configure(in_obj)
        self.out_shape = self.in_shape
        self.bias_size = self.in_shape[0]
        if self.weight_shape is None:
            self.weight_shape = (self.bias_size, 1)
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inference (bool): is inference only
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        if self.y is None or self.y.base is not self.outputs:
            self.y = self.outputs.reshape((self.bias_size, -1))
        self.y[:] = self.y + self.W
        return self.outputs

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas is None:
            self.deltas = error.reshape(self.y.shape)
        self.be.sum(self.deltas, axis=1, out=self.dW)
        return error


class Activation(Layer):

    """
    A layer that applies a specified transform to the inputs and
    produces outputs of the same shape.

    Generally used to implemenent nonlinearities for layer post activations.

    Arguments:
        transform (Transform): a transform object with fprop and bprop
            functions to apply
        name (str, optional): Layer name. Defaults to "ActivationLayer"
    """

    def __init__(self, transform, name=None):
        super(Activation, self).__init__(name)
        self.transform = transform
        self.owns_output = False
        self.owns_delta = True

    def __str__(self):
        return "Activation Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    @classmethod
    def gen_class(cls, pdict):
        assert 'transform' in pdict
        cname = pdict['transform']['type']
        tcls = load_class(cname)
        # many activations have no args
        if 'config' not in pdict['transform']:
            pdict['transform']['config'] = {}
        transf = tcls(**pdict['transform']['config'])
        pdict['transform'] = transf
        return cls(**pdict)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Activation, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        self.deltas[:] = self.transform.bprop(self.outputs) * error
        return self.deltas


class DataTransform(Layer):

    """
    A layer that applies a specified transform to input data in fprop only.

    Only supported as the first layer in the network.

    Arguments:
        transform (Transform): a transform object with fprop function to apply
        name (str, optional): Layer name. Defaults to "DataTransformLayer"
    """

    def __init__(self, transform, name=None):
        super(DataTransform, self).__init__(name)
        self.transform = transform
        self.owns_output = False

    def __str__(self):
        return "DataTransform Layer '%s': %s" % (
               self.name, self.transform.__class__.__name__)

    @classmethod
    def gen_class(cls, pdict):
        if pdict.get('transform') is not None:
            icls = load_class(pdict['transform']['type'])
            transform = icls(**pdict['transform']['config'])
            pdict['transform'] = transform
        return cls(**pdict)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(DataTransform, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        self.outputs[:] = self.transform(self.inputs)
        return self.outputs

    def bprop(self, *args):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            *args (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        return None


class CompoundLayer(list):
    """
    Base class for macro layers.
    """
    def __init__(self, bias=None, batch_norm=False, activation=None, name=None):
        super(CompoundLayer, self).__init__()
        if batch_norm and (bias is not None):
            raise AttributeError('Batchnorm and bias cannot be combined')
        self.activation = activation
        self.batch_norm = batch_norm
        self.bias = bias
        self.base_name = name

    @classmethod
    def gen_class(cls, pdict):
        for key in ['init', 'bias', 'activation']:
            if key in pdict and pdict[key] is not None:
                cname = pdict[key]['type']
                icls = load_class(cname)
                if 'config' not in pdict[key]:
                    pdict[key]['config'] = {}
                pdict[key] = icls(**pdict[key]['config'])
        return cls(**pdict)

    def init_base_name(self):
        if self.base_name is None:
            self.base_name = self[-1].name

    def add_postfilter_layers(self):
        self.init_base_name()
        if self.bias is not None:
            name = self.base_name + '_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name + '_bnorm'
            self.append(BatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Affine(CompoundLayer):

    """
    A linear layer with a learned bias and activation, implemented as a list
    composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, nout, init, bias=None,
                 batch_norm=False, activation=None, name=None,
                 parallelism="Disabled"):
        super(Affine, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Linear(nout, init, bsum=batch_norm, name=name,
                           parallelism=parallelism))
        self.add_postfilter_layers()


class BinaryAffine(CompoundLayer):

    """
    A binary linear layer with a learned bias and activation, implemented
    as a list composing separate linear, bias/batchnorm and activation layers.

    Arguments:
        nout (int, tuple): Desired size or shape of layer output
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, nout, init, bias=None,
                 batch_norm=False, activation=None, name=None):
        super(BinaryAffine, self).__init__(bias=bias, batch_norm=batch_norm,
                                           activation=activation, name=name)
        self.append(BinaryLinear(nout, init, bsum=batch_norm, name=name))
        self.add_postfilter_layers()

    def add_postfilter_layers(self):
        self.init_base_name()
        if self.bias is not None:
            name = self.base_name+'_bias'
            self.append(Bias(init=self.bias, name=name))
        if self.batch_norm:
            name = self.base_name+'_sbnorm'
            self.append(ShiftBatchNorm(name=name))
        if self.activation is not None:
            name = self.base_name + '_' + self.activation.classnm
            self.append(Activation(transform=self.activation, name=name))


class Conv(CompoundLayer):

    """
    A convolutional layer with a learned bias and activation, implemented as a
    list composing separate Convolution, Bias and Activation layers.

    Arguments:
        fshape (tuple(int)): three dimensional shape of convolution window
        init (Initializer, optional): Initializer object to use for
            initializing layer weights and bias
        strides (int, dict, optional): strides to apply convolution
            window over. An int applies to both dimensions, or a dict with
            str_h and str_w applies to h and w dimensions distinctly.  Defaults
            to str_w = str_h = None
        pad (int, dict, optional): padding to apply to edges of
            input. An int applies to both dimensions, or a dict with pad_h
            and pad_w applies to h and w dimensions distinctly.  Defaults
            to pad_w = pad_h = None
        bias (Initializer): an initializer to use for bias parameters
        activation (Transform): a transform object with fprop and bprop
            functions to apply
        name (str): the root name for the layer, suffixes are automatically
            generated for the component layers

    """

    def __init__(self, fshape, init, strides={}, padding={},
                 bias=None,
                 batch_norm=False,
                 activation=None,
                 name=None):
        super(Conv, self).__init__(bias=bias, batch_norm=batch_norm,
                                   activation=activation, name=name)
        self.append(Convolution(fshape=fshape, strides=strides, padding=padding,
                                init=init, bsum=batch_norm,
                                name=name))
        self.add_postfilter_layers()


class Deconv(CompoundLayer):

    """
    Same as Conv layer, but implements a composite deconvolution layer.
    """

    def __init__(self, fshape, init, strides={}, padding={}, bias=None, batch_norm=False,
                 activation=None, name=None):
        super(Deconv, self).__init__(bias=bias, batch_norm=batch_norm,
                                     activation=activation, name=name)
        self.append(Deconvolution(fshape=fshape, strides=strides, padding=padding,
                                  init=init, bsum=batch_norm))
        self.add_postfilter_layers()


class LRN(Layer):

    """
    Local Response Normalization layer.  This layer normalizes the output
    of each pixel/element across channels using the formula:

    .. math::

        output(h,w)_j = \\frac{output(h,w)_j}{(1 + (ascale/N) \sum{x(h,w)_i^2})^{bpower}}

    :math:`x(h,w)_i` is the input element at coordinate :math:`(h,w)` of the i-th feature map,
    :math:`output(h,w)_j` is the corresponding normalized output and the
    sum is taken over :math:`i` in the range :math:`[j - (depth-1)/2, j + (depth-1)/2]`

    Arguments:
        depth (int): the number of neighboring feature maps to include in
                     the normalization, depth must be odd and (depth-1)/2
                     neighbors are included from each side, zeros are added
                     as needed
        ascale (float): the normalization scaling factor (see equation above)
        bpower (float): the normalization exponent (see equation above)
        name (str): layer name

    """

    def __init__(self, depth, alpha=1., beta=0., ascale=1., bpower=1., name=None):
        super(LRN, self).__init__(name=name)
        self.J = depth
        self.depth = depth  # needed for serialization
        self.alpha = alpha
        self.beta = beta
        self.ascale = ascale
        self.bpower = bpower
        self.owns_delta = True
        self.lrnparams = {'J': self.J}
        self.nglayer = None

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(LRN, self).configure(in_obj)
        if self.nglayer is None:
            assert isinstance(self.in_shape, tuple)
            ikeys = ('C', 'H', 'W') if len(self.in_shape) == 3 else ('C', 'D', 'H', 'W')
            shapedict = {k: x for k, x in zip(ikeys, self.in_shape)}
            shapedict['N'] = self.be.bsz
            self.lrnparams.update(shapedict)
            self.nglayer = self.be.lrn_layer(self.be.default_dtype, **self.lrnparams)
            self.out_shape = self.in_shape
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(LRN, self).allocate(shared_outputs)
        self.denom = self.be.iobuf(self.in_shape)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.inputs = inputs
        self.be.fprop_lrn(self.nglayer,
                          inputs, self.outputs, self.denom,
                          self.alpha, self.beta, self.ascale, self.bpower)
        return self.outputs

    def bprop(self, error):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.deltas:
            self.be.bprop_lrn(self.nglayer,
                              self.inputs, self.outputs, error, self.deltas, self.denom,
                              self.alpha, self.beta, self.ascale, self.bpower)
        return self.deltas


class Dropout(Layer):

    """
    A dropout layer.

    Applies an element-wise multiplication of inputs with a keep mask.

    A keep mask is a tensor of ones and zeros of the same shape as the input.

    Each fprop call generates an new keep mask stochastically where there
    distribution of ones in the mask is controlled by the keep param.

    Arguments:
       keep (float): fraction of the inputs that should be stochastically kept.
    """

    def __init__(self, keep=0.5, name=None):
        super(Dropout, self).__init__(name)
        self.keep = keep
        self.keep_mask = None
        self.caffe_mode = self.be.check_caffe_compat()
        if self.caffe_mode:
            self._train_scaling = 1.0 / keep  # scaling factor during training
        else:
            self._train_scaling = 1.0  # override scaling factor to retain binary mask
        self.owns_output = False

    def __str__(self):
        return "Dropout Layer '%s': %d inputs and outputs, keep %d%% (caffe_compat %s)" % (
               self.name, self.nout, 100 * self.keep, self.caffe_mode)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(Dropout, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nout, _) = interpret_in_shape(self.in_shape)
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(Dropout, self).allocate(shared_outputs)
        self.keep_mask = self.be.iobuf(self.out_shape, parallelism=self.parallelism)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.outputs = self.inputs = inputs
        if inference:
            return self._fprop_inference(inputs)

        self.be.make_binary_mask(self.keep_mask, self.keep)
        self.outputs[:] = self.keep_mask * inputs * self._train_scaling

        return self.outputs

    def _fprop_inference(self, inputs):
        """
        Apply the forward pass transformation to the input data.

        May skip any computation not needed for doing inference only.

        Calling bprop subsequently is not valid.

        Arguments:
            inputs (Tensor): input data

        Returns:
            Tensor: output data
        """
        if not self.caffe_mode:
            self.outputs[:] = inputs * self.keep
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if not self.deltas:
            self.deltas = error
        self.deltas[:] = self.keep_mask * error * alpha * self._train_scaling + beta * error
        return self.deltas


class LookupTable(ParameterLayer):

    """
    A lookup table layer or a word embedding layer.

    The layer converts a word into a dense representation. When given a sentence,
    which is a vector of words (as integers), a matrix of vectors/embeddings for
    each word in the sentence is returned.

    LookupTable of dimensions embedding_dim by vocab_size is learnt.

    input shape - (nin, batch_size)

    output shape - (embedding_dim, nin * batch_size)

    weight shape - (embedding_dim, vocab_size)

    Arguments:
        vocab_size (int) : Number of words in the vocabulary
        embedding_dim (int) : Desired size of the word embedding
        init (Initializer): Initializer object to use for initializing layer weights
        name (str, optional): Layer name. Defaults to "LookupTableLayer"
    """

    def __init__(self, vocab_size, embedding_dim, init, update=True,
                 pad_idx=None, name=None):
        super(LookupTable, self).__init__(init, name)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.update = update
        self.pad_idx = pad_idx
        self.outputs_t = None

    def __str__(self):
        return "LookupTable Layer : %d inputs, (%d, %d) outputs size" % (
            self.nin, self.embedding_dim, self.nin)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(LookupTable, self).configure(in_obj)
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.out_shape = (self.embedding_dim, self.nin)
        if self.weight_shape is None:
            self.weight_shape = (self.vocab_size, self.embedding_dim)
        return self

    def allocate(self, shared_outputs=None):
        super(LookupTable, self).allocate()
        if self.inputs is None:
            self.inputs = self.be.zeros((1, self.nin * self.be.bsz),
                                        dtype=np.int32)  # inputs is np.float32
        self.dW[:] = 0
        if self.pad_idx is not None:
            self.W[self.pad_idx] = 0
        if self.outputs_t is None:
            self.outputs_t = self.be.empty_like(self.outputs.T)

    def fprop(self, inputs, inference=False):
        """
        Apply the forward pass transformation to the input data.

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        self.inputs[:] = inputs.reshape(self.inputs.shape)
        self.outputs_t[:] = self.W.take(self.inputs, axis=0)
        self.outputs[:] = self.outputs_t.T
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0):
        """
        Apply the backward pass transformation to the input data.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if self.update:
            self.dW[:] = 0
            self.be.compound_bprop_lut(self.nin, self.inputs, error, self.outputs_t,
                                       self.dW, self.pad_idx, alpha, beta)
        return self.deltas


class GeneralizedCost(NervanaObject):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

    def __init__(self, costfunc, name=None):
        super(GeneralizedCost, self).__init__(name)
        self.costfunc = costfunc
        self.outputs = None
        self.deltas = None
        self.cost_buffer = self.be.empty((1, 1))

    @classmethod
    def gen_class(cls, pdict):
        typ = pdict['costfunc']['type']
        ccls = load_class(typ)
        if 'config' not in pdict['costfunc']:
            pdict['costfunc']['config'] = {}
        pdict['costfunc'] = ccls.gen_class(pdict['costfunc']['config'])
        return cls(**pdict)

    def initialize(self, in_obj):
        """
        Determine dimensions of cost and error buffers and allocate space from the input layer

        Arguments:
            in_obj (Layer): input layer from which to calculate costs
        """

        assert isinstance(in_obj, Layer)
        self.prev_layer = in_obj
        self.parallelism = self.prev_layer.parallelism
        self.costfunc.parallelism = self.parallelism
        (_, self.nstep) = interpret_in_shape(in_obj.out_shape)
        self.outputs = self.be.iobuf((1, self.nstep),
                                     parallelism=self.parallelism,
                                     persist_values=False)
        self.deltas = self.be.iobuf(in_obj.out_shape,
                                    parallelism=self.parallelism,
                                    persist_values=False)
        self.cost = np.empty([1, 1], dtype=np.float32)

    def get_cost(self, inputs, targets):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor containing cost

        """
        self.outputs[:] = self.costfunc(inputs, targets)
        self.be.mean(self.outputs, axis=1, out=self.cost_buffer)
        self.cost = self.cost_buffer.get()
        return self.cost

    def get_errors(self, inputs, targets):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets (Tensor): Tensor containing target values.

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        self.deltas[:] = self.costfunc.bprop(inputs, targets)
        return self.deltas


class GeneralizedCostMask(GeneralizedCost):

    """
    A cost layer that applies the provided cost function and computes errors
    with respect to inputs and targets. Applies mask to deltas.

    Arguments:
       costfunc (Cost): class with costfunc that computes errors
    """

    def get_cost(self, inputs, targets_mask):
        """
        Compute the cost function over the inputs and targets.

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                targets
            targets_mask ((Tensor, Tensor)): Tuple with Tensor target values and Tensor mask

        Returns:
            Tensor containing cost
        """
        targets, mask = targets_mask
        masked_input = inputs * mask
        self.outputs[:] = self.costfunc(masked_input, targets)
        self.cost_buffer[:] = self.be.mean(self.outputs, axis=1)
        self.cost = self.cost_buffer.get()
        return self.cost

    def get_errors(self, inputs, targets_mask):
        """
        Compute the derivative of the cost function

        Arguments:
            inputs (Tensor): Tensor containing input values to be compared to
                             targets
            targets_mask ((Tensor, Tensor)): Tuple with Tensor target values
                                             and Tensor mask

        Returns:
            Tensor of same shape as the inputs containing their respective
            deltas.
        """
        targets, mask = targets_mask
        self.deltas[:] = self.costfunc.bprop(inputs, targets) * mask
        return self.deltas


class BatchNorm(Layer):

    """
    A batch normalization layer as described in [Ioffe2015]_.

    Normalizes a batch worth of inputs by subtracting batch mean and
    dividing by batch variance.  Then scales by learned factor gamma and
    shifts by learned bias beta.

    Uses the inputs to fprop to infer if a precomputed batch-sum is
    supplied from previous layer (input is tuple), or if the sum still
    needs to be computed.

    Notes:

    .. [Ioffe2015] http://arxiv.org/abs/1502.03167
    """

    def __init__(self, rho=0.9, eps=1e-3, name=None, binary=False):
        super(BatchNorm, self).__init__(name)
        self.allparams = None
        self.x = None  # used to point to reshaped view of inputs
        self.xhat = None
        self.has_params = True
        self.owns_delta = True
        self.error_view = None
        self.rho = rho
        self.eps = eps
        self.states = [[] for i in range(2)]
        self.relu = False
        self.beta = None
        self.gamma = None
        self.gmean = None
        self.gvar = None
        self.stats_dtype = np.float64 if self.be.default_dtype is np.float64 else np.float32
        self.binary = binary

    def __str__(self):
        return "BatchNorm Layer '%s': %d inputs, %d steps, %d feature maps" % (
               self.name, self.nin, self.nsteps, self.nfm)

    def configure(self, in_obj):
        """
        Sets shape based parameters of this layer given an input tuple or int
        or input layer.

        Arguments:
            in_obj (int, tuple, Layer or Tensor): object that provides shape
                                                  information for layer

        Returns:
            (tuple): shape of output data
        """
        super(BatchNorm, self).configure(in_obj)
        self.out_shape = self.in_shape
        (self.nin, self.nsteps) = interpret_in_shape(self.in_shape)
        self.nfm = self.in_shape[0] if isinstance(self.in_shape, tuple) else self.nin
        return self

    def allocate(self, shared_outputs=None):
        """
        Allocate output buffer to store activations from fprop.

        Arguments:
            shared_outputs (Tensor, optional): pre-allocated tensor for activations to be
                                               computed into
        """
        super(BatchNorm, self).allocate(shared_outputs)
        self.y = self.outputs.reshape((self.nfm, -1))
        self.xvar = self.be.zeros((self.nfm, 1), dtype=self.stats_dtype)
        if self.allparams is None:
            self.init_params(self.nfm)
        if self.prev_layer in (None, True) or not getattr(self.prev_layer, 'batch_sum', None):
            self.xsum = self.be.zeros((self.nfm, 1),
                                      dtype=self.stats_dtype, **self.get_param_attrs())
            self.xsum.auto_reduce = False
            self.compute_batch_sum = True
        else:
            self.xsum = self.prev_layer.batch_sum
            self.compute_batch_sum = False

    def init_params(self, dim0):
        self.beta = self.be.zeros((dim0, 1), dtype=self.stats_dtype, **self.get_param_attrs())
        self.gamma = self.be.ones((dim0, 1), dtype=self.stats_dtype, **self.get_param_attrs())
        self.params = [self.beta, self.gamma]

        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        self.inf_params = [self.be.zeros_like(p) for p in self.params]

        (self.grad_beta, self.grad_gamma) = self.grad_params
        (self.gmean, self.gvar) = self.inf_params

        self.allparams = self.params + self.inf_params

    @property
    def plist(self):
        return [((p, g), s) for p, g, s in zip(self.params, self.grad_params, self.states)]

    def fprop(self, inputs, inference=False, beta=0.0):
        """
        Normalize inputs (x) over batch mean and variance.
        xhat = (x - xmean) / xvar

        Scale and shift normalized inputs (xhat) by learned parameters gamma and beta.
        y = xhat * gamma + beta

        Accumulate partial results to global mean and variance buffers used for inference.

        Arguments:
            inputs:
            inference:  (Default value = False)
            beta:  (Default value = 0.0)

        Returns:
            Tensor: output data
        """
        if self.inputs is None or self.inputs.base is not inputs:
            self.inputs = inputs.reshape((self.nfm, -1))

        if inference:
            return self._fprop_inference(self.inputs, beta)

        if self.compute_batch_sum:
            self.xsum[:] = self.be.sum(self.inputs, axis=1)

        self.be.compound_fprop_bn(
            self.inputs, self.xsum, self.xvar, self.gmean, self.gvar, self.gamma,
            self.beta, self.outputs, self.eps, self.rho, beta, self.relu,
            binary=self.binary)

        return self.outputs

    def _fprop_inference(self, inputs, beta=0.0):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        xhat = (inputs - self.gmean) / self.be.sqrt(self.gvar + self.eps)  # Op-tree only
        self.y[:] = self.y * beta + xhat * self.gamma + self.beta
        return self.outputs

    def bprop(self, error, alpha=1.0, beta=0.0):
        """
        Compute gradients for learning gamma and beta as well as layer weights.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer
            alpha (float, optional): scale to apply to input for activation
                                     gradient bprop.  Defaults to 1.0
            beta (float, optional): scale to apply to output activation
                                    gradient bprop.  Defaults to 0.0

        Returns:

        """
        assert alpha == 1.0 and beta == 0.0
        if not self.error_view:
            self.error_view = error.reshape((self.nfm, -1))

        self.be.compound_bprop_bn(self.deltas, self.grad_gamma, self.grad_beta,
                                  self.error_view,
                                  self.inputs, self.xsum, self.xvar, self.gamma,
                                  self.eps, binary=self.binary)
        return self.deltas

    def get_params(self):
        return self.plist

    def get_params_serialize(self, keep_states=True):
        return self.get_description(get_weights=True, keep_states=keep_states)

    def get_description(self, get_weights=False, keep_states=True):
        """
        Get layer parameters.

        Arguments:
            get_weights (bool, optional): Control whether all parameters are returned or
                                          just weights for serialization.
            keep_states (bool, optional): Controls whether the states should be returned
        """
        serial_dict = super(BatchNorm, self).get_description()
        if get_weights:
            serial_dict['params'] = {}
            for key in ['beta', 'gamma', 'gmean', 'gvar']:
                serial_dict['params'][key] = getattr(self, key).get()

            if keep_states:
                serial_dict['states'] = [[s.get() for s in slist] for slist in self.states]
        return serial_dict

    def set_params(self, pdict):
        if type(pdict['params']) is dict:
            for key, val in pdict['params'].items():
                if isinstance(getattr(self, key), Tensor):
                    getattr(self, key).set(val)
                else:
                    setattr(self, key, self.be.array(val, **self.get_param_attrs()))

            self.params = [self.beta, self.gamma]
            self.inf_params = [self.gmean, self.gvar]
            self.allparams = self.params + self.inf_params
        else:
            logger.error('Using old serialization format.  This will be'
                         ' deprecated in future release. Resave serialized file'
                         ' using current format')

            self.allparams = [self.be.array(x, **self.get_param_attrs()) for x in pdict['params']]
            self.params = self.allparams[:2]
            self.inf_params = self.allparams[2:]
            (self.beta, self.gamma) = self.params
            (self.gmean, self.gvar) = self.inf_params

        self.grad_params = [self.be.zeros_like(p) for p in self.params]
        (self.grad_beta, self.grad_gamma) = self.grad_params

    def set_states(self, pdict):
        if not any(self.states):
            self.states = [[self.be.array(x, **self.get_param_attrs()) for x in slist]
                           for slist in pdict['states']]
        else:
            for dlist, slist in zip(self.states, pdict['states']):
                for dst, src in zip(dlist, slist):
                    dst.set(src)


class BatchNormAutodiff(BatchNorm):

    """
    An example to use autodiff in batchnorm.
    """

    def __init__(self, rho=0.99, eps=1e-6, name=None):
        super(BatchNormAutodiff, self).__init__(rho, eps, name)

    def get_forward_optree(self):
        """
        Initialize the fprop optree for batchnorm.
        """
        # get fprop op-tree
        xvar = self.be.var(self.x, axis=1)
        xmean = self.be.mean(self.x, axis=1)
        xhat = (self.x - xmean) / self.be.sqrt(xvar + self.eps)
        return xhat * self.gamma + self.beta

    def fprop(self, inputs, inference=False):
        """
        Compute the actual fprop from op-tree, update the global estimations

        Arguments:
            inputs (Tensor): input data
            inference (bool): is inference only

        Returns:
            Tensor: output data
        """
        if inference:
            return self._fprop_inference(inputs)
        self.init_buffers(inputs)
        if self.allparams is None:
            self.init_params(self.nfm)
            self.fprop_op_tree = self.get_forward_optree()

        # the actual f-prop
        self.y[:] = self.fprop_op_tree

        # for inference
        self.gmean[:] = (self.gmean * self.rho + (1.0 - self.rho) * self.be.mean(self.x, axis=1))
        self.gvar[:] = (self.gvar * self.rho + (1.0 - self.rho) * self.be.var(self.x, axis=1))

        return self.outputs

    def bprop(self, error):
        """
        Use Autodiff.back_prop_grad to back propagate gradients for the
        corresponding tensors.

        Arguments:
            error (Tensor): deltas back propagated from the adjacent higher layer

        Returns:
            Tensor: deltas to propagate to the adjacent lower layer
        """
        if not self.deltas:
            self.deltas = error.reshape((self.nfm, -1))

        # autodiff will automatically cache and reuse the object
        # if we know the `error` buffer at init, we can also create the autodiff
        # object at layer's init
        ad = Autodiff(self.fprop_op_tree, self.be, next_error=self.deltas)

        # back propagate
        ad.back_prop_grad([self.x, self.gamma, self.beta],
                          [self.deltas, self.grad_gamma, self.grad_beta])

        return error


class ShiftBatchNorm(BatchNorm):

    """
    Shift based batch normalization.

    Reference:
        http://arxiv.org/pdf/1602.02830v3.pdf
    """
    def __init__(self, rho=0.99, eps=1e-6, name=None):
        super(ShiftBatchNorm, self).__init__(rho, eps, name, binary=True)

    def _fprop_inference(self, inputs, beta=0.0):
        """
        Apply one linear transformation that captures normalization, gamma scaling and beta shift.
        """
        input_ms = self.be.empty_like(inputs)
        input_ms[:] = inputs - self.gmean
        inv_v = self.be.empty_like(self.gvar)
        inv_v[:] = 1.0 / self.be.sqrt(self.gvar + self.eps)
        xhat = self.be.shift(input_ms, inv_v)
        self.y[:] = self.y * beta + self.be.shift(xhat, self.gamma) + self.beta
        return self.outputs
