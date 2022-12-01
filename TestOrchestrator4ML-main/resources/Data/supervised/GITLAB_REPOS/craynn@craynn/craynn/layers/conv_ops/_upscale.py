import tensorflow as tf
from craygraph import derive

from ..meta import Layer, model_from

from .conv_utils import *

__all__ = [
  'Upscale2DLayer',
  'BilinearUpscale2DLayer',
  'BicubicUpscale2DLayer',

  'upscale_2d',
  'bilinear_upscale_2d',
  'bicubic_upscale',

  'upscale',
  'bilinear_upscale',
  'bicubic_upscale_2d'
]

class UpscaleLayer(Layer):
  def __init__(self, incoming, ndim, kernel_size=2, mode='nearest', name=None):
    assert ndim == 2, 'only 2D upsampling is supported'

    self.ndim = ndim
    self.kernel_size = get_kernel_size(kernel_size, ndim)
    self.mode = mode.lower()

    super(UpscaleLayer, self).__init__(incoming, name=name)

  def get_output_for(self, incoming):
    incoming_shape = tf.shape(incoming)
    new_size = tuple(incoming_shape[i + 2] * self.kernel_size[i] for i in range(self.ndim))

    transposed = to_channel_last(incoming, ndim=self.ndim)

    resized = tf.image.resize(
      transposed, size=new_size, method=get_upscale_mode(self.mode)
    )

    return to_channel_fist(resized, ndim=self.ndim)

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], input_shape[1]) + tuple(
      int(size * factor)
      for size, factor in zip(get_spatial_dims(input_shape), self.kernel_size)
    )

Upscale2DLayer = derive('Upscale2DLayer').based_on(UpscaleLayer).with_fixed(ndim=2, mode='nearest')
BilinearUpscale2DLayer = derive('Upscale2DLayer').based_on(UpscaleLayer).with_fixed(ndim=2, mode='bilinear')
BicubicUpscale2DLayer = derive('Upscale2DLayer').based_on(UpscaleLayer).with_fixed(ndim=2, mode='bicubic')

upscale_2d = model_from(Upscale2DLayer)()
bilinear_upscale_2d = model_from(BilinearUpscale2DLayer)()
bicubic_upscale_2d = model_from(BicubicUpscale2DLayer)()

upscale = upscale_2d
bilinear_upscale = bilinear_upscale_2d
bicubic_upscale = bicubic_upscale_2d