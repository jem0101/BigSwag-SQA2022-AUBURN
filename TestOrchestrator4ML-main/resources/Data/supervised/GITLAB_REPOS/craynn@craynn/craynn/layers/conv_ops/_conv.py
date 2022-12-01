import tensorflow as tf

from craygraph import derive

from ...parameters import default_weight_init, default_bias_init
from ...nonlinearities import default_nonlinearity

from ..common import Layer
from ..meta import get_output_shape, model_from

from .conv_utils import *

__all__ = [
  'Conv1DLayer', 'Conv2DLayer', 'Conv3DLayer',
  'PaddedConv1DLayer', 'PaddedConv2DLayer', 'PaddedConv3DLayer',
  'Conv1DLayer1x1', 'Conv2DLayer1x1', 'Conv3DLayer1x1',

  'conv_1d', 'conv_2d', 'conv_3d', 'conv',
  'pconv_1d', 'pconv_2d', 'pconv_3d', 'pconv',
  'conv_1d_1x1', 'conv_2d_1x1', 'conv_3d_1x1', 'conv_1x1',

  'DepthwiseConvLayer2D', 'depthwise_conv_2d'
]

class ConvLayer(Layer):
  def __init__(
    self, incoming, ndim,
    num_filters,
    kernel_size=3,
    activation=default_nonlinearity,
    padding='valid', stride=1, dilation=1,
    W=default_weight_init,
    b=default_bias_init,
    name=None
  ):
    self.ndim = ndim

    input_shape = get_output_shape(incoming)

    if len(input_shape) != self.ndim + 2:
      raise ValueError(
        '%dD conv layer accepts only %dD tensors [got %s]!' % (self.ndim, self.ndim + 2, input_shape)
      )

    if num_filters <= 0:
      raise ValueError(
        '`num` channels must be > 0!'
      )

    self.input_channels = get_channel_dim(input_shape)
    self.input_size = get_spatial_dims(input_shape)
    self.num_channels = num_filters

    self.kernel_size = get_kernel_size(kernel_size, ndim=ndim)
    self.kernel_shape = self.kernel_size + (self.input_channels, self.num_channels)

    self.dilation = get_kernel_size(dilation, ndim=ndim)
    self.stride = get_kernel_size(stride, ndim=ndim)

    self.padding = padding.lower()

    self.activation = activation

    self.conv_op = self._get_conv_op(ndim)

    self.W = W(self.kernel_shape, name='W', weights=True, conv_kernel=True, trainable=True)
    self.b = b((self.num_channels,), name='b', biases=True, trainable=True)

    super(ConvLayer, self).__init__(
      incoming,
      name=name,
      parameters=(self.W, self.b),
    )

  def _get_conv_op(self, ndim):
    return get_conv_op(ndim)

  def get_output_for(self, W, b, input, channel_last=False):
    if channel_last:
      input = to_channel_last(input, ndim=self.ndim)

    convolved = self.conv_op(
      input, W,
      strides=self.stride,
      padding=get_padding(self.padding),
      data_format=get_data_format(self.ndim, channel_last=channel_last),
      dilations=self.dilation,
      name=str(self) + '_conv%dd' % self.ndim,
    )

    if channel_last:
      convolved = to_channel_fist(convolved, self.ndim)

    broadcast = (None, ) + (slice(None), ) + (None, )* self.ndim
    return self.activation(
       convolved + b[broadcast]
    )

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    if len(input_shape) != self.ndim + 2:
      raise ValueError(
        '%dD conv layer accepts only %dD tensors [got %s]!' % (self.ndim, self.ndim + 2, input_shape)
      )

    return (input_shape[0], self.num_channels) + tuple(
      get_conv_output_shape(
        input_shape[i + 2],
        self.kernel_size[i],
        stride=self.stride[i],
        dilation=self.dilation[i],
        pad=self.padding
      )
      for i in range(self.ndim)
    )


Conv1DLayer = derive('Conv1DLayer').based_on(ConvLayer).with_fixed(ndim=1)
Conv2DLayer = derive('Conv2DLayer').based_on(ConvLayer).with_fixed(ndim=2)
Conv3DLayer = derive('Conv3DLayer').based_on(ConvLayer).with_fixed(ndim=3)

conv_1d = model_from(Conv1DLayer)()
conv_2d = model_from(Conv2DLayer)()
conv_3d = model_from(Conv3DLayer)()

conv = select_by_dim([None, conv_1d, conv_2d, conv_3d])

Conv1DLayer1x1 = derive('Conv1DLayer1x1').based_on(ConvLayer).with_fixed(ndim=1, kernel_size=1)
Conv2DLayer1x1 = derive('Conv2DLayer1x1').based_on(ConvLayer).with_fixed(ndim=2, kernel_size=1)
Conv3DLayer1x1 = derive('Conv3DLayer1x1').based_on(ConvLayer).with_fixed(ndim=3, kernel_size=1)

conv_1d_1x1 = model_from(Conv1DLayer1x1)()
conv_2d_1x1 = model_from(Conv2DLayer1x1)()
conv_3d_1x1 = model_from(Conv3DLayer1x1)()

conv_1x1 = select_by_dim([None, conv_1d_1x1, conv_2d_1x1, conv_3d_1x1])

conv_1d = model_from(Conv1DLayer)()
conv_2d = model_from(Conv2DLayer)()
conv_3d = model_from(Conv3DLayer)()


PaddedConv1DLayer = derive('PaddedConv1DLayer').based_on(Conv1DLayer).with_fixed(padding='same')
PaddedConv2DLayer = derive('PaddedConv2DLayer').based_on(Conv2DLayer).with_fixed(padding='same')
PaddedConv3DLayer = derive('PaddedConv3DLayer').based_on(Conv3DLayer).with_fixed(padding='same')

pconv_1d = model_from(PaddedConv1DLayer)()
pconv_2d = model_from(PaddedConv2DLayer)()
pconv_3d = model_from(PaddedConv3DLayer)()

pconv = select_by_dim([None, pconv_1d, pconv_2d, pconv_3d])

class DepthwiseConvLayer2D(ConvLayer):
  def __init__(self,
     incoming,
     num_filters,
     kernel_size=3,
     activation=default_nonlinearity,
     padding='valid', stride=1, dilation=1,
     W=default_weight_init,
     b=default_bias_init,
     name=None
  ):
    super(DepthwiseConvLayer2D, self).__init__(
      incoming,
      ndim=2, num_filters=num_filters,
      kernel_size=kernel_size,
      activation=activation,
      padding=padding, stride=stride, dilation=dilation,
      W=W, b=b, name=name
    )

  def _get_conv_op(self, ndim):
    return tf.nn.depthwise_conv2d

  def get_output_shape_for(self, W_shape, b_shape, input_shape):
    if len(input_shape) != 4:
      raise ValueError(
        '2D depthwise conv layer accepts only 2D tensors [got %s]!' % (input_shape, )
      )

    return (input_shape[0], input_shape[1] * self.num_channels) + tuple(
      get_conv_output_shape(
        input_shape[i + 2],
        self.kernel_size[i],
        stride=self.stride[i],
        dilation=self.dilation[i],
        pad=self.padding
      )
      for i in range(self.ndim)
    )

depthwise_conv_2d = model_from(DepthwiseConvLayer2D)()