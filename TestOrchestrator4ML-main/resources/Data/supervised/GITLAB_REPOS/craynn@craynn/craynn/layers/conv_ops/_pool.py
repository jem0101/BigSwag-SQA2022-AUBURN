import tensorflow as tf
from craygraph import derive

from ..meta import Layer, model_from
from .conv_utils import *

__all__ = [
  'GeneralPooling1DLayer', 'GeneralPooling2DLayer', 'GeneralPooling3DLayer',
  'MaxPool1DLayer', 'MaxPool2DLayer', 'MaxPool3DLayer',
  'MeanPool1DLayer', 'MeanPool2DLayer', 'MeanPool3DLayer',

  'general_pool_1d', 'general_pool_2d', 'general_pool_3d', 'general_pool',
  'mean_pool_1d', 'mean_pool_2d', 'mean_pool_3d', 'mean_pool',
  'max_pool_1d', 'max_pool_2d', 'max_pool_3d', 'max_pool',

  'GlobalPoolLayer', 'global_pool',
    
  'GlobalMaxPool1D', 'GlobalMaxPool2D', 'GlobalMaxPool3D',
  'GlobalMeanPool1D', 'GlobalMeanPool2D', 'GlobalMeanPool3D',

  'global_max_pool_1d', 'global_max_pool_2d', 'global_max_pool_3d', 'global_max_pool',
  'global_mean_pool_1d', 'global_mean_pool_2d', 'global_mean_pool_3d', 'global_mean_pool'
]

class PoolingLayer(Layer):
  def __init__(self, incoming,
               ndim=None,
               kernel_size=2,
               pooling_function='max',
               pad='valid',
               stride=None,
               name=None):
    self.ndim = ndim

    self.pooling_op = get_pool_op(pooling_function, ndim)
    self.pooling_function = pooling_function.lower()

    self.kernel_size = get_kernel_size(kernel_size, ndim=ndim)

    if stride is None:
      stride = self.kernel_size

    self.stride = get_kernel_size(stride, ndim=ndim)
    self.pad = pad.lower()

    super(PoolingLayer, self).__init__(incoming, name=name)

  def get_output_for(self, input, channel_last=False):
    if channel_last:
      input = to_channel_last(input, ndim=self.ndim)

    pooled = self.pooling_op(
      input,
      ksize=self.kernel_size,
      strides=self.stride,
      padding=get_padding(self.pad),
      data_format=get_data_format(self.ndim, channel_last=channel_last),
      name=str(self) + '_pool%dd' % self.ndim,
    )

    if channel_last:
      pooled = to_channel_fist(pooled, self.ndim)

    return pooled

  def get_output_shape_for(self, input_shape):
    if len(input_shape) != self.ndim + 2:
      raise ValueError(
        '%dD pool layer accepts only %dD tensors [got %s]!' % (input_shape, self.ndim, len(input_shape))
      )

    return (input_shape[0], input_shape[1]) + tuple([
      get_conv_output_shape(
        input_shape[i + 2],
        self.kernel_size[i],
        stride=self.stride[i],
        dilation=1,
        pad=self.pad
      )
      for i in range(self.ndim)
    ])


GeneralPooling1DLayer = derive('GeneralPooling1D').based_on(PoolingLayer).with_fixed(ndim=1)
GeneralPooling2DLayer = derive('GeneralPooling2D').based_on(PoolingLayer).with_fixed(ndim=2)
GeneralPooling3DLayer = derive('GeneralPooling3D').based_on(PoolingLayer).with_fixed(ndim=3)

general_pool_1d = model_from(GeneralPooling1DLayer)()
general_pool_2d = model_from(GeneralPooling2DLayer)()
general_pool_3d = model_from(GeneralPooling3DLayer)()

general_pool = select_by_dim([None, general_pool_1d, general_pool_2d, general_pool_3d])

MaxPool1DLayer = derive('MaxPool1DLayer').based_on(PoolingLayer).with_fixed(ndim=1, pooling_function='max', stride=None)
MaxPool2DLayer = derive('MaxPool2DLayer').based_on(PoolingLayer).with_fixed(ndim=2, pooling_function='max', stride=None)
MaxPool3DLayer = derive('MaxPool3DLayer').based_on(PoolingLayer).with_fixed(ndim=3, pooling_function='max', stride=None)

max_pool_1d = model_from(MaxPool1DLayer)()
max_pool_2d = model_from(MaxPool2DLayer)()
max_pool_3d = model_from(MaxPool3DLayer)()

max_pool = select_by_dim([None, max_pool_1d, max_pool_2d, max_pool_3d])

MeanPool1DLayer = derive('MeanPool1DLayer').based_on(PoolingLayer).with_fixed(ndim=1, pooling_function='mean', stride=None)
MeanPool2DLayer = derive('MeanPool2DLayer').based_on(PoolingLayer).with_fixed(ndim=2, pooling_function='mean', stride=None)
MeanPool3DLayer = derive('MeanPool3DLayer').based_on(PoolingLayer).with_fixed(ndim=3, pooling_function='mean', stride=None)

mean_pool_1d = model_from(MeanPool1DLayer)()
mean_pool_2d = model_from(MeanPool2DLayer)()
mean_pool_3d = model_from(MeanPool3DLayer)()

mean_pool = select_by_dim([None, mean_pool_1d, mean_pool_2d, mean_pool_3d])


class GlobalPoolLayer(Layer):
  def __init__(self, incoming, pool_f, axis, name=None):
    self.pool_f = pool_f
    self.axis = axis

    super(GlobalPoolLayer, self).__init__(incoming, name=name)

  def get_output_for(self, incoming):
    return self.pool_f(incoming, axis=self.axis)

  def get_output_shape_for(self, incoming_shape):
    return tuple(
      incoming_shape[i]
      for i in range(len(incoming_shape))
      if i not in self.axis
    )

global_pool = model_from(GlobalPoolLayer)()

GlobalMaxPool1D = derive('GlobalMaxPool1DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_max, axis=(2, ))
GlobalMaxPool2D = derive('GlobalMaxPool2DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_max, axis=(2, 3))
GlobalMaxPool3D = derive('GlobalMaxPool3DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_max, axis=(2, 3, 4))

global_max_pool_1d = model_from(GlobalMaxPool1D)()
global_max_pool_2d = model_from(GlobalMaxPool2D)()
global_max_pool_3d = model_from(GlobalMaxPool3D)()

global_max_pool = select_by_dim([None, global_max_pool_1d, global_max_pool_2d, global_max_pool_3d])

GlobalMeanPool1D = derive('GlobalMeanPool1DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_mean, axis=(2, ))
GlobalMeanPool2D = derive('GlobalMeanPool2DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_mean, axis=(2, 3))
GlobalMeanPool3D = derive('GlobalMeanPool3DLayer').based_on(GlobalPoolLayer).with_fixed(pool_f=tf.reduce_mean, axis=(2, 3, 4))

global_mean_pool_1d = model_from(GlobalMeanPool1D)()
global_mean_pool_2d = model_from(GlobalMeanPool2D)()
global_mean_pool_3d = model_from(GlobalMeanPool3D)()

global_mean_pool = select_by_dim([None, global_mean_pool_1d, global_mean_pool_2d, global_mean_pool_3d])