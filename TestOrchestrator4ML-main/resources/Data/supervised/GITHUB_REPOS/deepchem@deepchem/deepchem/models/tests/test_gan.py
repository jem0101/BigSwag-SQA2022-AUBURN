import deepchem as dc
import numpy as np
import tensorflow as tf
import unittest
import tempfile
from tensorflow.keras.layers import Input, Concatenate, Dense
from flaky import flaky


def generate_batch(batch_size):
  """Draw training data from a Gaussian distribution, where the mean  is a conditional input."""
  means = 10 * np.random.random([batch_size, 1])
  values = np.random.normal(means, scale=2.0)
  return means, values


def generate_data(gan, batches, batch_size):
  for i in range(batches):
    means, values = generate_batch(batch_size)
    batch = {gan.data_inputs[0]: values, gan.conditional_inputs[0]: means}
    yield batch


class ExampleGAN(dc.models.GAN):

  def get_noise_input_shape(self):
    return (2,)

  def get_data_input_shapes(self):
    return [(1,)]

  def get_conditional_input_shapes(self):
    return [(1,)]

  def create_generator(self):
    noise_input = Input(self.get_noise_input_shape())
    conditional_input = Input(self.get_conditional_input_shapes()[0])
    inputs = [noise_input, conditional_input]
    gen_in = Concatenate(axis=1)(inputs)
    output = Dense(1)(gen_in)
    return tf.keras.Model(inputs=inputs, outputs=output)

  def create_discriminator(self):
    data_input = Input(self.get_data_input_shapes()[0])
    conditional_input = Input(self.get_conditional_input_shapes()[0])
    inputs = [data_input, conditional_input]
    discrim_in = Concatenate(axis=1)(inputs)
    dense = Dense(10, activation=tf.nn.relu)(discrim_in)
    output = Dense(1, activation=tf.sigmoid)(dense)
    return tf.keras.Model(inputs=inputs, outputs=output)


@flaky
def test_cgan():
  """Test fitting a conditional GAN."""

  gan = ExampleGAN(learning_rate=0.01)
  gan.fit_gan(
      generate_data(gan, 500, 100), generator_steps=0.5, checkpoint_interval=0)

  # See if it has done a plausible job of learning the distribution.

  means = 10 * np.random.random([1000, 1])
  values = gan.predict_gan_generator(conditional_inputs=[means])
  deltas = values - means
  assert abs(np.mean(deltas)) < 1.0
  assert np.std(deltas) > 1.0
  assert gan.get_global_step() == 500


@flaky
def test_cgan_reload():
  """Test reloading a conditional GAN."""

  model_dir = tempfile.mkdtemp()
  gan = ExampleGAN(learning_rate=0.01, model_dir=model_dir)
  gan.fit_gan(generate_data(gan, 500, 100), generator_steps=0.5)

  # See if it has done a plausible job of learning the distribution.
  means = 10 * np.random.random([1000, 1])
  batch_size = len(means)
  noise_input = gan.get_noise_batch(batch_size=batch_size)
  values = gan.predict_gan_generator(
      noise_input=noise_input, conditional_inputs=[means])
  deltas = values - means
  assert abs(np.mean(deltas)) < 1.0
  assert np.std(deltas) > 1.0
  assert gan.get_global_step() == 500

  reloaded_gan = ExampleGAN(learning_rate=0.01, model_dir=model_dir)
  reloaded_gan.restore()
  reloaded_values = reloaded_gan.predict_gan_generator(
      noise_input=noise_input, conditional_inputs=[means])

  assert np.all(values == reloaded_values)


@flaky
def test_mix_gan_reload():
  """Test reloading a GAN with multiple generators and discriminators."""

  model_dir = tempfile.mkdtemp()
  gan = ExampleGAN(
      n_generators=2,
      n_discriminators=2,
      learning_rate=0.01,
      model_dir=model_dir)
  gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.5)

  reloaded_gan = ExampleGAN(
      n_generators=2,
      n_discriminators=2,
      learning_rate=0.01,
      model_dir=model_dir)
  reloaded_gan.restore()
  # See if it has done a plausible job of learning the distribution.

  means = 10 * np.random.random([1000, 1])
  batch_size = len(means)
  noise_input = gan.get_noise_batch(batch_size=batch_size)
  for i in range(2):
    values = gan.predict_gan_generator(
        noise_input=noise_input, conditional_inputs=[means], generator_index=i)
    reloaded_values = reloaded_gan.predict_gan_generator(
        noise_input=noise_input, conditional_inputs=[means], generator_index=i)
    assert np.all(values == reloaded_values)
  assert gan.get_global_step() == 1000
  # No training has been done after reload
  assert reloaded_gan.get_global_step() == 0


@flaky
def test_mix_gan():
  """Test a GAN with multiple generators and discriminators."""

  gan = ExampleGAN(n_generators=2, n_discriminators=2, learning_rate=0.01)
  gan.fit_gan(
      generate_data(gan, 1000, 100), generator_steps=0.5, checkpoint_interval=0)

  # See if it has done a plausible job of learning the distribution.

  means = 10 * np.random.random([1000, 1])
  for i in range(2):
    values = gan.predict_gan_generator(
        conditional_inputs=[means], generator_index=i)
    deltas = values - means
    assert abs(np.mean(deltas)) < 1.0
    assert np.std(deltas) > 1.0
  assert gan.get_global_step() == 1000


@flaky
def test_wgan():
  """Test fitting a conditional WGAN."""

  class ExampleWGAN(dc.models.WGAN):

    def get_noise_input_shape(self):
      return (2,)

    def get_data_input_shapes(self):
      return [(1,)]

    def get_conditional_input_shapes(self):
      return [(1,)]

    def create_generator(self):
      noise_input = Input(self.get_noise_input_shape())
      conditional_input = Input(self.get_conditional_input_shapes()[0])
      inputs = [noise_input, conditional_input]
      gen_in = Concatenate(axis=1)(inputs)
      output = Dense(1)(gen_in)
      return tf.keras.Model(inputs=inputs, outputs=output)

    def create_discriminator(self):
      data_input = Input(self.get_data_input_shapes()[0])
      conditional_input = Input(self.get_conditional_input_shapes()[0])
      inputs = [data_input, conditional_input]
      discrim_in = Concatenate(axis=1)(inputs)
      dense = Dense(10, activation=tf.nn.relu)(discrim_in)
      output = Dense(1)(dense)
      return tf.keras.Model(inputs=inputs, outputs=output)

  # We have to set the gradient penalty very small because the generator's
  # output is only a single number, so the default penalty would constrain
  # it far too much.

  gan = ExampleWGAN(learning_rate=0.01, gradient_penalty=0.1)
  gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.1)

  # See if it has done a plausible job of learning the distribution.

  means = 10 * np.random.random([1000, 1])
  values = gan.predict_gan_generator(conditional_inputs=[means])
  deltas = values - means
  assert abs(np.mean(deltas)) < 1.0
  assert np.std(deltas) > 1.0


@flaky
def test_wgan_reload():
  """Test fitting a conditional WGAN."""

  class ExampleWGAN(dc.models.WGAN):

    def get_noise_input_shape(self):
      return (2,)

    def get_data_input_shapes(self):
      return [(1,)]

    def get_conditional_input_shapes(self):
      return [(1,)]

    def create_generator(self):
      noise_input = Input(self.get_noise_input_shape())
      conditional_input = Input(self.get_conditional_input_shapes()[0])
      inputs = [noise_input, conditional_input]
      gen_in = Concatenate(axis=1)(inputs)
      output = Dense(1)(gen_in)
      return tf.keras.Model(inputs=inputs, outputs=output)

    def create_discriminator(self):
      data_input = Input(self.get_data_input_shapes()[0])
      conditional_input = Input(self.get_conditional_input_shapes()[0])
      inputs = [data_input, conditional_input]
      discrim_in = Concatenate(axis=1)(inputs)
      dense = Dense(10, activation=tf.nn.relu)(discrim_in)
      output = Dense(1)(dense)
      return tf.keras.Model(inputs=inputs, outputs=output)

  # We have to set the gradient penalty very small because the generator's
  # output is only a single number, so the default penalty would constrain
  # it far too much.

  model_dir = tempfile.mkdtemp()
  gan = ExampleWGAN(
      learning_rate=0.01, gradient_penalty=0.1, model_dir=model_dir)
  gan.fit_gan(generate_data(gan, 1000, 100), generator_steps=0.1)

  reloaded_gan = ExampleWGAN(
      learning_rate=0.01, gradient_penalty=0.1, model_dir=model_dir)
  reloaded_gan.restore()

  # See if it has done a plausible job of learning the distribution.
  means = 10 * np.random.random([1000, 1])
  batch_size = len(means)
  noise_input = gan.get_noise_batch(batch_size=batch_size)
  values = gan.predict_gan_generator(
      noise_input=noise_input, conditional_inputs=[means])
  reloaded_values = reloaded_gan.predict_gan_generator(
      noise_input=noise_input, conditional_inputs=[means])
  assert np.all(values == reloaded_values)
