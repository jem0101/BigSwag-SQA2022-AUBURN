import numpy as np

__all__ = [
  'downsample',
  'resize'
]

def downsample(*factors):
  def f(imgs):
    from skimage.transform import downscale_local_mean
    new_shape = (imgs.shape[1] // factors[0], imgs.shape[2] // factors[1])
    result = np.ndarray(shape=(imgs.shape[0], ) + new_shape + (imgs.shape[3], ), dtype=imgs.dtype)

    for i in range(imgs.shape[0]):
      result[i] = downscale_local_mean(imgs[i], factors=factors + (1, ))

    return result

  return f

def resize(new_size, order=3, dtype='float32', **kwargs):
  def f(imgs):
    from skimage.transform import resize

    result = np.ndarray(shape=(imgs.shape[0], ) + new_size + (imgs.shape[3], ), dtype=dtype)

    for i in range(imgs.shape[0]):
      result[i] = resize(imgs[i].astype(dtype), output_shape=new_size, order=order, **kwargs)

    return result

  return f