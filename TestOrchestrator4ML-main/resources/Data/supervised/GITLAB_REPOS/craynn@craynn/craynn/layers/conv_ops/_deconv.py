from craygraph import derive

from ...nonlinearities import default_nonlinearity
from ...parameters import default_weight_init, default_bias_init

from ..common import Layer
from ..meta import get_output_shape, model_from

from .conv_utils import *

__all__ = [
  'TransposedConv2DLayer',

  'transposed_conv_2d',
  'deconv_2d',
  'transposed_conv',
  'deconv',

  'ptransposed_conv_2d',
  'pdeconv_2d',
  'ptransposed_conv',
  'pdeconv',
]

class TransposedConvLayer(Layer):
  def __init__(self, incoming, ndim,
               num_filters,
               kernel_size=3,
               activation=default_nonlinearity,
               pad='valid', stride=1,
               W=default_weight_init,
               b=default_bias_init,
               name=None):
    self.ndim = ndim
    self.op = get_deconv_op(ndim)

    input_shape = get_output_shape(incoming)

    if len(input_shape) != self.ndim + 2:
      raise ValueError(
        '%dD deconv layer accepts only %dD tensors [got %s]!' % (self.ndim, self.ndim + 2, input_shape)
      )

    if num_filters <= 0:
      raise ValueError(
        '`num` channels must be > 0!'
      )

    self.input_channels = get_channel_dim(input_shape)
    self.input_size = get_spatial_dims(input_shape)
    self.num_channels = num_filters

    self.kernel_size = get_kernel_size(kernel_size, ndim=ndim)
    self.kernel_shape = self.kernel_size + (self.num_channels, self.input_channels)

    self.stride = get_kernel_size(stride, ndim=ndim)

    self.pad = pad.lower()
    self.activation = activation

    self.W = W(self.kernel_shape, name='W', weights=True, conv_kernel=True, trainable=True)
    self.b = b((self.num_channels,), name='b', biases=True, trainable=True)

    super(TransposedConvLayer, self).__init__(
      incoming,
      name=name,
      parameters=(self.W, self.b)
    )

  def get_output_for(self, W, b, input, channel_last=False):
    if channel_last:
      input = to_channel_last(input, ndim=self.ndim)

    ### apparently, tf.shape(x) and X.shape give different results...
    output_shape = self.get_output_shape_for(W.shape, b.shape, input.shape)

    convolved = self.op(
      input, W,
      output_shape=output_shape,
      strides=self.stride,
      padding=get_padding(self.pad),
      data_format=get_data_format(self.ndim, channel_last=channel_last),
      name=str(self) + 'deconv%dd' % self.ndim,
    )

    if channel_last:
      convolved = to_channel_fist(convolved, self.ndim)

    broadcast = (None,) + (slice(None),) + (None,) * self.ndim
    return self.activation(
      convolved + b[broadcast]
    )

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    if len(input_shape) != self.ndim + 2:
      raise ValueError(
        '%dD deconv layer accepts only %dD tensors [got %s]!' % (self.ndim, self.ndim + 2, input_shape)
      )

    return tuple([input_shape[0], self.num_channels] + [
      get_deconv_output_shape(
        input_shape[i + 2],
        kernel_size=self.kernel_size[i],
        stride=self.stride[i],
        pad=self.pad
      )
      for i in range(self.ndim)
    ])


TransposedConv1DLayer = derive('TransposedConv1DLayer').based_on(TransposedConvLayer).with_fixed(ndim=1)
TransposedConv2DLayer = derive('TransposedConv2DLayer').based_on(TransposedConvLayer).with_fixed(ndim=2)
TransposedConv3DLayer = derive('TransposedConv3DLayer').based_on(TransposedConvLayer).with_fixed(ndim=3)

transposed_conv_1d = model_from(TransposedConv1DLayer)()
deconv_1d = transposed_conv_1d

transposed_conv_2d = model_from(TransposedConv2DLayer)()
deconv_2d = transposed_conv_2d

transposed_conv_3d = model_from(TransposedConv3DLayer)()
deconv_3d = transposed_conv_3d

transposed_conv = select_by_dim([None, transposed_conv_1d, transposed_conv_2d, transposed_conv_3d])
deconv = transposed_conv

PaddedTransposedConv1DLayer = derive('PaddedTransposedConv1DLayer').based_on(TransposedConvLayer).with_fixed(ndim=1, pad='same')
PaddedTransposedConv2DLayer = derive('PaddedTransposedConv2DLayer').based_on(TransposedConvLayer).with_fixed(ndim=2, pad='same')
PaddedTransposedConv3DLayer = derive('PaddedTransposedConv3DLayer').based_on(TransposedConvLayer).with_fixed(ndim=3, pad='same')

ptransposed_conv_1d = model_from(PaddedTransposedConv1DLayer)()
pdeconv_1d = ptransposed_conv_1d

ptransposed_conv_2d = model_from(PaddedTransposedConv2DLayer)()
pdeconv_2d = ptransposed_conv_2d

ptransposed_conv_3d = model_from(PaddedTransposedConv3DLayer)()
pdeconv_3d = ptransposed_conv_3d

ptransposed_conv = select_by_dim([None, ptransposed_conv_1d, ptransposed_conv_2d, ptransposed_conv_3d])
pdeconv = ptransposed_conv