import pytest

def get_cpu_tf():
  import tensorflow as tf
  tf.config.experimental.set_visible_devices([], 'GPU')
  return tf

def get_gpu_tf():
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')

  if len(gpus) > 0:
    # Currently, memory growth needs to be the same across GPUs
    # for gpu in gpus:
    #   tf.config.experimental.set_memory_growth(gpu, True)

    tf.config.experimental.set_virtual_device_configuration(
      gpus[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
    )
  else:
    raise Exception('There is no GPUs.')

  return tf

@pytest.fixture(scope="session")
def gpu_tf():
  yield get_gpu_tf()

@pytest.fixture(scope="session")
def cpu_tf():
  yield get_cpu_tf()

@pytest.fixture(scope="session")
def tf():
  try:
    yield get_gpu_tf()
  except:
    yield get_cpu_tf()