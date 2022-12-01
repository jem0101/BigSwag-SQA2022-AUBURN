import tensorflow as tf

from ..meta import get_output_shape, model_selector

__all__ = [
  'get_kernel_size',
  'get_data_format',
  'get_padding',
  'get_pool_mode',
  'get_upscale_mode',

  'get_conv_op',
  'get_deconv_op',
  'get_pool_op',

  'get_channel_dim',
  'get_spatial_dims',

  'get_conv_output_shape',
  'get_deconv_output_shape',

  'to_channel_last',
  'to_channel_fist',

  'select_by_dim'
]

DATA_FORMATS = {
  1 : 'NCW',
  2 : 'NCHW',
  3 : 'NCDHW'
}

CHANNEL_LAST_DATA_FORMATS = {
  1 : 'NWC',
  2 : 'NHWC',
  3 : 'NDHWC'
}


def to_channel_last(tensor, ndim=None):
  if ndim is None:
    ndim = len(tf.shape(tensor)) - 2

  permutation = (0, ) + tuple(range(2, ndim + 2)) + (1, )
  return tf.transpose(tensor, permutation)

def to_channel_fist(tensor, ndim=None):
  if ndim is None:
    ndim = len(tf.shape(tensor)) - 2

  permutation = (0, ndim + 1) + tuple(range(1, ndim + 1))
  return tf.transpose(tensor, permutation)

def get_data_format(ndim, channel_last=False):
  if ndim in DATA_FORMATS:
    if channel_last:
      return CHANNEL_LAST_DATA_FORMATS[ndim]
    else:
      return DATA_FORMATS[ndim]
  else:
    raise ValueError('%d-dim instances format is not supported' % ndim)

POOL_MODES = {
  'max' : 'MAX',
  'mean' : 'AVG'
}

def get_pool_mode(mode):
  if mode is POOL_MODES:
    return POOL_MODES[mode]
  else:
    raise ValueError('Unknown pool mode %s' % (mode, ))

PADDINGS = {
  'valid' : 'VALID',
  'same' : 'SAME'
}

def get_padding(pad):
  if pad in PADDINGS:
    return PADDINGS[pad]
  else:
    return pad

UPSCALE_MODES = {
  'nearest' : tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  'bilinear' : tf.image.ResizeMethod.BILINEAR,
  'bicubic' : tf.image.ResizeMethod.BICUBIC,
}

def get_upscale_mode(mode):
  if mode in UPSCALE_MODES:
    return UPSCALE_MODES[mode]
  else:
    raise ValueError('Unknown upscale mode %s' % (mode, ))

def conv1d(input, filters, strides, padding, data_format='NCW', dilations=None, name=None):
  return tf.nn.conv1d(
    input=input,
    filters=filters,
    stride=strides,
    padding=padding,
    data_format=data_format,
    dilations=dilations,
    name=name
  )

CONV_OPS = {
  1 : conv1d,
  2 : tf.nn.conv2d,
  3 : tf.nn.conv3d,
}

def get_conv_op(ndim):
  if ndim in CONV_OPS:
    return CONV_OPS[ndim]
  else:
    raise ValueError('%d-dim conv is not supported' % ndim)

DECONV_OPS = {
  1 : tf.nn.conv1d_transpose,
  2 : tf.nn.conv2d_transpose,
  3 : tf.nn.conv3d_transpose
}

def get_deconv_op(ndim):
  if ndim in DECONV_OPS:
    return DECONV_OPS[ndim]
  else:
    raise ValueError('%d-dim deconv is not supported' % ndim)

POOL_OPS = {
  ('max', 1) : tf.nn.max_pool1d,
  ('max', 2) : tf.nn.max_pool2d,
  ('max', 3) : tf.nn.max_pool3d,

  ('mean', 1) : tf.nn.avg_pool1d,
  ('mean', 2) : tf.nn.avg_pool2d,
  ('mean', 3) : tf.nn.avg_pool3d,
}

def get_pool_op(mode, ndim):
  if (mode, ndim) in POOL_OPS:
    return POOL_OPS[(mode, ndim)]
  else:
    raise ValueError('Uknown pooling operation: %s %dd' % (mode, ndim))


def get_kernel_size(kernel_size, ndim):
  if type(kernel_size) is int:
    return (kernel_size, ) * ndim
  else:
    try:
      if all([ type(n) for n in kernel_size ]) and len(kernel_size) == ndim:
        return tuple(kernel_size)
    except:
      raise ValueError('kernel_size is neither iterable, nor int [%s]' % kernel_size)

get_channel_dim = lambda shape: shape[1]
get_spatial_dims = lambda shape: shape[2:]

def int_ceil(a, b):
  return (a + b - 1) // b

def get_conv_output_shape(input_shape, kernel_size, stride=1, pad='valid', dilation=1):
  if input_shape is None:
    return None

  effective_kernel_size = (kernel_size - 1) * dilation + 1

  pad = pad.lower()

  if pad == 'same':
    effective_input_shape = input_shape
  elif pad == 'valid':
    effective_input_shape = input_shape - effective_kernel_size + 1
  else:
    raise ValueError('Unknown pad [%s]' % pad)

  return int_ceil(effective_input_shape, stride)

def get_deconv_output_shape(input_shape, kernel_size, stride=1, pad='valid'):
  if input_shape is None:
    return None

  pad = pad.lower()

  if pad == 'same':
    delta = kernel_size // 2
  elif pad == 'valid':
    delta = 0
  else:
    raise ValueError('Unknown pad [%s]' % pad)

  return (input_shape - 1) * stride - 2 * delta + kernel_size

@model_selector
def select_by_dim(models):
  def selector(incoming):
    ndim = len(get_output_shape(incoming)) - 2
    return models[ndim]

  return selector