__author__ = "Joseph Gomes"
__copyright__ = "Copyright 2017, Stanford University"
__license__ = "MIT"

import sys

from deepchem.models import KerasModel
from deepchem.models.layers import AtomicConvolution
from deepchem.models.losses import L2Loss
from tensorflow.keras.layers import Input, Layer

import numpy as np
import tensorflow as tf
import itertools


def initializeWeightsBiases(prev_layer_size,
                            size,
                            weights=None,
                            biases=None,
                            name=None):
  """Initializes weights and biases to be used in a fully-connected layer.

  Parameters
  ----------
  prev_layer_size: int
    Number of features in previous layer.
  size: int
    Number of nodes in this layer.
  weights: tf.Tensor, optional (Default None)
    Weight tensor.
  biases: tf.Tensor, optional (Default None)
    Bias tensor.
  name: str
    Name for this op, optional (Defaults to 'fully_connected' if None)

  Returns
  -------
  weights: tf.Variable
    Initialized weights.
  biases: tf.Variable
    Initialized biases.

  """

  if weights is None:
    weights = tf.random.truncated_normal([prev_layer_size, size], stddev=0.01)
  if biases is None:
    biases = tf.zeros([size])

  w = tf.Variable(weights, name='w')
  b = tf.Variable(biases, name='b')
  return w, b


class AtomicConvScore(Layer):
  """The scoring function used by the atomic convolution models."""

  def __init__(self, atom_types, layer_sizes, **kwargs):
    super(AtomicConvScore, self).__init__(**kwargs)
    self.atom_types = atom_types
    self.layer_sizes = layer_sizes

  def build(self, input_shape):
    self.type_weights = []
    self.type_biases = []
    self.output_weights = []
    self.output_biases = []
    n_features = int(input_shape[0][-1])
    layer_sizes = self.layer_sizes
    num_layers = len(layer_sizes)
    weight_init_stddevs = [1 / np.sqrt(x) for x in layer_sizes]
    bias_init_consts = [0.0] * num_layers
    for ind, atomtype in enumerate(self.atom_types):
      prev_layer_size = n_features
      self.type_weights.append([])
      self.type_biases.append([])
      self.output_weights.append([])
      self.output_biases.append([])
      for i in range(num_layers):
        weight, bias = initializeWeightsBiases(
            prev_layer_size=prev_layer_size,
            size=layer_sizes[i],
            weights=tf.random.truncated_normal(
                shape=[prev_layer_size, layer_sizes[i]],
                stddev=weight_init_stddevs[i]),
            biases=tf.constant(
                value=bias_init_consts[i], shape=[layer_sizes[i]]))
        self.type_weights[ind].append(weight)
        self.type_biases[ind].append(bias)
        prev_layer_size = layer_sizes[i]
      weight, bias = initializeWeightsBiases(prev_layer_size, 1)
      self.output_weights[ind].append(weight)
      self.output_biases[ind].append(bias)

  def call(self, inputs):
    frag1_layer, frag2_layer, complex_layer, frag1_z, frag2_z, complex_z = inputs
    atom_types = self.atom_types
    num_layers = len(self.layer_sizes)

    def atomnet(current_input, atomtype):
      prev_layer = current_input
      for i in range(num_layers):
        layer = tf.nn.bias_add(
            tf.matmul(prev_layer, self.type_weights[atomtype][i]),
            self.type_biases[atomtype][i])
        layer = tf.nn.relu(layer)
        prev_layer = layer

      output_layer = tf.squeeze(
          tf.nn.bias_add(
              tf.matmul(prev_layer, self.output_weights[atomtype][0]),
              self.output_biases[atomtype][0]))
      return output_layer

    frag1_zeros = tf.zeros_like(frag1_z, dtype=tf.float32)
    frag2_zeros = tf.zeros_like(frag2_z, dtype=tf.float32)
    complex_zeros = tf.zeros_like(complex_z, dtype=tf.float32)

    frag1_atomtype_energy = []
    frag2_atomtype_energy = []
    complex_atomtype_energy = []

    for ind, atomtype in enumerate(atom_types):
      frag1_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag1_layer)
      frag2_outputs = tf.map_fn(lambda x: atomnet(x, ind), frag2_layer)
      complex_outputs = tf.map_fn(lambda x: atomnet(x, ind), complex_layer)

      cond = tf.equal(frag1_z, atomtype)
      frag1_atomtype_energy.append(tf.where(cond, frag1_outputs, frag1_zeros))
      cond = tf.equal(frag2_z, atomtype)
      frag2_atomtype_energy.append(tf.where(cond, frag2_outputs, frag2_zeros))
      cond = tf.equal(complex_z, atomtype)
      complex_atomtype_energy.append(
          tf.where(cond, complex_outputs, complex_zeros))

    frag1_outputs = tf.add_n(frag1_atomtype_energy)
    frag2_outputs = tf.add_n(frag2_atomtype_energy)
    complex_outputs = tf.add_n(complex_atomtype_energy)

    frag1_energy = tf.reduce_sum(frag1_outputs, 1)
    frag2_energy = tf.reduce_sum(frag2_outputs, 1)
    complex_energy = tf.reduce_sum(complex_outputs, 1)
    binding_energy = complex_energy - (frag1_energy + frag2_energy)
    return tf.expand_dims(binding_energy, axis=1)


class AtomicConvModel(KerasModel):
  """Implements an Atomic Convolution Model.

  Implements the atomic convolutional networks as introduced in

  Gomes, Joseph, et al. "Atomic convolutional networks for predicting protein-ligand binding affinity." arXiv preprint arXiv:1703.10603 (2017).

  The atomic convolutional networks function as a variant of
  graph convolutions. The difference is that the "graph" here is
  the nearest neighbors graph in 3D space. The AtomicConvModel
  leverages these connections in 3D space to train models that
  learn to predict energetic state starting from the spatial
  geometry of the model.
  """

  def __init__(self,
               frag1_num_atoms=70,
               frag2_num_atoms=634,
               complex_num_atoms=701,
               max_num_neighbors=12,
               batch_size=24,
               atom_types=[
                   6, 7., 8., 9., 11., 12., 15., 16., 17., 20., 25., 30., 35.,
                   53., -1.
               ],
               radial=[[
                   1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
                   7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
               ], [0.0, 4.0, 8.0], [0.4]],
               layer_sizes=[32, 32, 16],
               learning_rate=0.001,
               **kwargs):
    """
    Parameters
    ----------
    frag1_num_atoms: int
      Number of atoms in first fragment
    frag2_num_atoms: int
      Number of atoms in sec
    max_num_neighbors: int
      Maximum number of neighbors possible for an atom. Recall neighbors
      are spatial neighbors.
    atom_types: list
      List of atoms recognized by model. Atoms are indicated by their
      nuclear numbers.
    radial: list
      TODO: add description
    layer_sizes: list
      TODO: add description
    learning_rate: float
      Learning rate for the model.
    """
    # TODO: Turning off queue for now. Safe to re-activate?
    self.complex_num_atoms = complex_num_atoms
    self.frag1_num_atoms = frag1_num_atoms
    self.frag2_num_atoms = frag2_num_atoms
    self.max_num_neighbors = max_num_neighbors
    self.batch_size = batch_size
    self.atom_types = atom_types

    rp = [x for x in itertools.product(*radial)]
    frag1_X = Input(shape=(frag1_num_atoms, 3))
    frag1_nbrs = Input(shape=(frag1_num_atoms, max_num_neighbors))
    frag1_nbrs_z = Input(shape=(frag1_num_atoms, max_num_neighbors))
    frag1_z = Input(shape=(frag1_num_atoms,))

    frag2_X = Input(shape=(frag2_num_atoms, 3))
    frag2_nbrs = Input(shape=(frag2_num_atoms, max_num_neighbors))
    frag2_nbrs_z = Input(shape=(frag2_num_atoms, max_num_neighbors))
    frag2_z = Input(shape=(frag2_num_atoms,))

    complex_X = Input(shape=(complex_num_atoms, 3))
    complex_nbrs = Input(shape=(complex_num_atoms, max_num_neighbors))
    complex_nbrs_z = Input(shape=(complex_num_atoms, max_num_neighbors))
    complex_z = Input(shape=(complex_num_atoms,))

    self._frag1_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([frag1_X, frag1_nbrs, frag1_nbrs_z])

    self._frag2_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([frag2_X, frag2_nbrs, frag2_nbrs_z])

    self._complex_conv = AtomicConvolution(
        atom_types=self.atom_types, radial_params=rp,
        boxsize=None)([complex_X, complex_nbrs, complex_nbrs_z])

    score = AtomicConvScore(self.atom_types, layer_sizes)([
        self._frag1_conv, self._frag2_conv, self._complex_conv, frag1_z,
        frag2_z, complex_z
    ])

    model = tf.keras.Model(
        inputs=[
            frag1_X, frag1_nbrs, frag1_nbrs_z, frag1_z, frag2_X, frag2_nbrs,
            frag2_nbrs_z, frag2_z, complex_X, complex_nbrs, complex_nbrs_z,
            complex_z
        ],
        outputs=score)
    super(AtomicConvModel, self).__init__(
        model, L2Loss(), batch_size=batch_size, **kwargs)

  def default_generator(self,
                        dataset,
                        epochs=1,
                        mode='fit',
                        deterministic=True,
                        pad_batches=True):
    batch_size = self.batch_size

    def replace_atom_types(z):

      def place_holder(i):
        if i in self.atom_types:
          return i
        return -1

      return np.array([place_holder(x) for x in z])

    for epoch in range(epochs):
      for ind, (F_b, y_b, w_b, ids_b) in enumerate(
          dataset.iterbatches(
              batch_size, deterministic=True, pad_batches=pad_batches)):
        N = self.complex_num_atoms
        N_1 = self.frag1_num_atoms
        N_2 = self.frag2_num_atoms
        M = self.max_num_neighbors

        batch_size = F_b.shape[0]
        num_features = F_b[0][0].shape[1]
        frag1_X_b = np.zeros((batch_size, N_1, num_features))
        for i in range(batch_size):
          frag1_X_b[i] = F_b[i][0]

        frag2_X_b = np.zeros((batch_size, N_2, num_features))
        for i in range(batch_size):
          frag2_X_b[i] = F_b[i][3]

        complex_X_b = np.zeros((batch_size, N, num_features))
        for i in range(batch_size):
          complex_X_b[i] = F_b[i][6]

        frag1_Nbrs = np.zeros((batch_size, N_1, M))
        frag1_Z_b = np.zeros((batch_size, N_1))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][2])
          frag1_Z_b[i] = z
        frag1_Nbrs_Z = np.zeros((batch_size, N_1, M))
        for atom in range(N_1):
          for i in range(batch_size):
            atom_nbrs = F_b[i][1].get(atom, "")
            frag1_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag1_Nbrs_Z[i, atom, j] = frag1_Z_b[i, atom_j]

        frag2_Nbrs = np.zeros((batch_size, N_2, M))
        frag2_Z_b = np.zeros((batch_size, N_2))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][5])
          frag2_Z_b[i] = z
        frag2_Nbrs_Z = np.zeros((batch_size, N_2, M))
        for atom in range(N_2):
          for i in range(batch_size):
            atom_nbrs = F_b[i][4].get(atom, "")
            frag2_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              frag2_Nbrs_Z[i, atom, j] = frag2_Z_b[i, atom_j]

        complex_Nbrs = np.zeros((batch_size, N, M))
        complex_Z_b = np.zeros((batch_size, N))
        for i in range(batch_size):
          z = replace_atom_types(F_b[i][8])
          complex_Z_b[i] = z
        complex_Nbrs_Z = np.zeros((batch_size, N, M))
        for atom in range(N):
          for i in range(batch_size):
            atom_nbrs = F_b[i][7].get(atom, "")
            complex_Nbrs[i, atom, :len(atom_nbrs)] = np.array(atom_nbrs)
            for j, atom_j in enumerate(atom_nbrs):
              complex_Nbrs_Z[i, atom, j] = complex_Z_b[i, atom_j]

        inputs = [
            frag1_X_b, frag1_Nbrs, frag1_Nbrs_Z, frag1_Z_b, frag2_X_b,
            frag2_Nbrs, frag2_Nbrs_Z, frag2_Z_b, complex_X_b, complex_Nbrs,
            complex_Nbrs_Z, complex_Z_b
        ]
        y_b = np.reshape(y_b, newshape=(batch_size, 1))
        yield (inputs, [y_b], [w_b])
