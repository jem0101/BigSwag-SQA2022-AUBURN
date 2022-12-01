# -*- coding: utf-8 -*-

r'''Feedforward layers for neural network computation graphs.'''

from __future__ import division

import numpy as np
import theano.sparse as SS
import theano.tensor as TT

from . import base
from .. import util

__all__ = [
    'Classifier',
    'Feedforward',
    'Tied',
]


class Feedforward(base.Layer):
    '''A feedforward neural network layer performs a transform of its input.

    More precisely, feedforward layers as implemented here perform an affine
    transformation of their input, followed by a potentially nonlinear
    :ref:`activation function <activations>` performed elementwise on the
    transformed input.

    Feedforward layers are the fundamental building block on which most neural
    network models are built.

    Notes
    -----

    This layer can be constructed using the forms ``'feedforward'`` or ``'ff'``.

    *Parameters*

    - With one input:

      - ``b`` --- bias
      - ``w`` --- weights

    - With :math:`N>1` inputs:

      - ``b`` --- bias
      - ``w_1`` --- weight for input 1
      - ``w_2`` ...
      - ``w_N`` --- weight for input :math:`N`

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer
    '''

    __extra_registration_keys__ = ['ff']

    def _weight_for_input(self, name):
        return 'w' if len(self._input_shapes) == 1 else 'w_{}'.format(name)

    def transform(self, inputs):
        def _dot(x, y):
            if isinstance(x, SS.SparseVariable):
                return SS.structured_dot(x, y)
            else:
                return TT.dot(x, y)

        xws = ((inputs[name], self.find(self._weight_for_input(name)))
               for name in self._input_shapes)
        pre = sum(_dot(x, w) for x, w in xws) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def setup(self):
        for name, shape in self._input_shapes.items():
            label = self._weight_for_input(name)
            self.add_weights(label, shape[-1], self.output_size)
        self.add_bias('b', self.output_size)


class Classifier(Feedforward):
    '''A classifier layer performs a softmax over a linear input transform.

    Classifier layers are typically the "output" layer of a classifier network.

    This layer type really only wraps the output activation of a standard
    :class:`Feedforward` layer.

    Notes
    -----

    The classifier layer is just a vanilla :class:`Feedforward` layer that uses
    a ``'softmax'`` output :ref:`activation <activations>`.
    '''

    __extra_registration_keys__ = ['softmax']

    def __init__(self, **kwargs):
        kwargs['activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)


class Tied(base.Layer):
    '''A tied-weights feedforward layer shadows weights from another layer.

    Notes
    -----

    Tied weights are typically featured in some types of autoencoder models
    (e.g., PCA). A layer with tied weights requires a "partner" layer -- the
    tied layer borrows the weights from its partner and uses the transpose of
    them to perform its feedforward mapping. Thus, tied layers do not have their
    own weights. On the other hand, tied layers do have their own bias values,
    but these can be fixed to zero during learning to simulate networks with no
    bias (e.g., PCA on mean-centered data).

    *Parameters*

    - ``b`` --- bias

    *Outputs*

    - ``out`` --- the post-activation state of the layer
    - ``pre`` --- the pre-activation state of the layer

    Parameters
    ----------
    partner : str or :class:`theanets.layers.base.Layer`
        The "partner" layer to which this layer is tied.

    Attributes
    ----------
    partner : :class:`theanets.layers.base.Layer`
        The "partner" layer to which this layer is tied.
    '''

    def __init__(self, partner, **kwargs):
        self.partner = partner
        kwargs['size'] = kwargs['shape'] = None
        if isinstance(partner, base.Layer):
            kwargs['shape'] = partner.input_shape
        super(Tied, self).__init__(**kwargs)

    def transform(self, inputs):
        x = inputs[self.input_name]
        pre = TT.dot(x, self.partner.find('w').T) + self.find('b')
        return dict(pre=pre, out=self.activate(pre)), []

    def resolve_inputs(self, layers):
        super(Tied, self).resolve_inputs(layers)
        if isinstance(self.partner, util.basestring):
            # if the partner is named, just get that layer.
            matches = [l for l in layers if l.name == self.partner]
            if len(matches) != 1:
                raise util.ConfigurationError(
                    'tied layer "{}": cannot find partner "{}"'
                    .format(self.name, self.partner))
            self.partner = matches[0]

    def resolve_outputs(self):
        self._output_shapes['out'] = self.partner.input_shape

    def setup(self):
        # this layer does not create a weight matrix!
        self.add_bias('b', self.output_size)

    def log(self):
        inputs = ', '.join('"{0}" {1}'.format(*ns) for ns in self._input_shapes.items())
        util.log('layer {0.__class__.__name__} "{0.name}" '
                 '(tied to "{0.partner.name}") {0.output_shape} {1} from {2}',
                 self, getattr(self.activate, 'name', self.activate), inputs)
        util.log('learnable parameters: {}', self.log_params())

    def to_spec(self):
        spec = super(Tied, self).to_spec()
        spec['partner'] = self.partner.name
        return spec
