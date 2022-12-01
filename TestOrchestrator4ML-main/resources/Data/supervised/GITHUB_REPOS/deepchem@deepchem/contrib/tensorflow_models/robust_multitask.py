from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import warnings
import numpy as np
import tensorflow as tf

try:
  from collections.abc import Sequence as SequenceCollection
except:
  from collections import Sequence as SequenceCollection

from deepchem.nn import model_ops

class RobustMultitaskClassifier(MultiTaskClassifier):
  """Implements a neural network for robust multitasking.
  
  Key idea is to have bypass layers that feed directly from features
  to task output. Hopefully will allow tasks to route around bad
  multitasking.
  """

  def __init__(self,
               n_tasks,
               n_features,
               layer_sizes=[1000],
               weight_init_stddevs=0.02,
               bypass_layer_sizes=[100],
               bypass_weight_init_stddevs=[.02],
               bypass_bias_init_consts=[1.],
               bypass_dropouts=[.5],
               **kwargs):
    """Create a MultiTaskClassifier.

    In addition to the following arguments, this class also accepts
    all the keyword arguments from MultiTaskClassifier.

    Parameters
    ----------
    n_tasks: int
      number of tasks
    n_features: int
      number of features
    layer_sizes: list
      the size of each dense layer in the network.  The length of
      this list determines the number of layers.
    weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight
      initialization of each layer.  The length of this list should
      equal len(layer_sizes).  Alternatively this may be a single
      value instead of a list, in which case the same value is used
      for every layer.
    bypass_layer_sizes: list
      the size of each dense bypass layer in the network. The length
      of this list determines the number of layers.
    bypass_weight_init_stddevs: list or float
      the standard deviation of the distribution to use for weight
      initialization of each layer.  The length of this list should
      equal len(bypass_layer_sizes).  Alternatively this may be a
      single value instead of a list, in which case the same value is
      used for every layer.
    bypass_bias_init_consts: list or loat
      the value to initialize the biases in each layer to.  The
      length of this list should equal len(bypass_layer_sizes).
      Alternatively this may be a single value instead of a list, in
      which case the same value is used for every layer.
    bypass_dropouts: list or float
      the dropout probablity to use for each layer.  The length of
      this list should equal len(bypass_layer_sizes).  Alternatively
      this may be a single value instead of a list, in which case the
      same value is used for every layer.
    """
    self.bypass_layer_sizes = bypass_layer_sizes
    self.bypass_weight_init_stddevs = bypass_weight_init_stddevs
    self.bypass_bias_init_consts = bypass_bias_init_consts
    self.bypass_dropouts = bypass_dropouts

    n_layers = len(layer_sizes)
    assert n_layers == len(bypass_layer_sizes)
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bypass_weight_init_stddevs, SequenceCollection):
      bypass_weight_init_stddevs = [bypass_weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers

    # Add the input features.
    mol_features = Feature(shape=(None, n_features))
    prev_layer = mol_features


      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts

      bypass_layer_sizes = self.bypass_layer_sizes
      bypass_weight_init_stddevs = self.bypass_weight_init_stddevs
      bypass_bias_init_consts = self.bypass_bias_init_consts
      bypass_dropouts = self.bypass_dropouts

      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      assert len(lengths_set) == 1, "All layer params must have same length."
      num_layers = lengths_set.pop()
      assert num_layers > 0, "Must have some layers defined."

      bypass_lengths_set = {
          len(bypass_layer_sizes),
          len(bypass_weight_init_stddevs),
          len(bypass_bias_init_consts),
          len(bypass_dropouts),
      }
      assert len(bypass_lengths_set) == 1, (
          "All bypass_layer params" + " must have same length.")
      num_bypass_layers = bypass_lengths_set.pop()

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue(
            [mol_features] + label_placeholders + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        prev_layer = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        prev_layer = mol_features

      top_layer = prev_layer
      prev_layer_size = num_features
      for i in range(num_layers):
        # layer has shape [None, layer_sizes[i]]
        print("Adding weights of shape %s" % str(
            [prev_layer_size, layer_sizes[i]]))
        layer = tf.nn.relu(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, layer_sizes[i]],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(
                    value=bias_init_consts[i], shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      # top_multitask_layer has shape [None, layer_sizes[-1]]
      top_multitask_layer = prev_layer
      for task in range(self.n_tasks):
        # TODO(rbharath): Might want to make it feasible to have multiple
        # bypass layers.
        # Construct task bypass layer
        prev_bypass_layer = top_layer
        prev_bypass_layer_size = num_features
        for i in range(num_bypass_layers):
          # bypass_layer has shape [None, bypass_layer_sizes[i]]
          print("Adding bypass weights of shape %s" % str(
              [prev_bypass_layer_size, bypass_layer_sizes[i]]))
          bypass_layer = tf.nn.relu(
              model_ops.fully_connected_layer(
                  tensor=prev_bypass_layer,
                  size=bypass_layer_sizes[i],
                  weight_init=tf.truncated_normal(
                      shape=[prev_bypass_layer_size, bypass_layer_sizes[i]],
                      stddev=bypass_weight_init_stddevs[i]),
                  bias_init=tf.constant(
                      value=bypass_bias_init_consts[i],
                      shape=[bypass_layer_sizes[i]])))

          bypass_layer = model_ops.dropout(bypass_layer, bypass_dropouts[i],
                                           training)
          prev_bypass_layer = bypass_layer
          prev_bypass_layer_size = bypass_layer_sizes[i]
        top_bypass_layer = prev_bypass_layer

        if num_bypass_layers > 0:
          # task_layer has shape [None, layer_sizes[-1] + bypass_layer_sizes[-1]]
          task_layer = tf.concat(
              axis=1, values=[top_multitask_layer, top_bypass_layer])
          task_layer_size = layer_sizes[-1] + bypass_layer_sizes[-1]
        else:
          task_layer = top_multitask_layer
          task_layer_size = layer_sizes[-1]
        print("Adding output weights of shape %s" % str([task_layer_size, 1]))
        output.append(
            model_ops.logits(
                task_layer,
                num_classes=2,
                weight_init=tf.truncated_normal(
                    shape=[task_layer_size, 2], stddev=weight_init_stddevs[-1]),
                bias_init=tf.constant(value=bias_init_consts[-1], shape=[2])))
      return (output, labels, weights)


class RobustMultitaskRegressor(TensorflowMultiTaskRegressor):
  """Implements a neural network for robust multitasking.
  
  Key idea is to have bypass layers that feed directly from features to task
  output. Hopefully will allow tasks to route around bad multitasking.
  """

  def __init__(self,
               n_tasks,
               n_features,
               logdir=None,
               bypass_layer_sizes=[100],
               bypass_weight_init_stddevs=[.02],
               bypass_bias_init_consts=[1.],
               bypass_dropouts=[.5],
               **kwargs):
    warnings.warn("RobustMultiTaskRegressor is deprecated. "
                  "Will be removed in DeepChem 1.4.", DeprecationWarning)
    self.bypass_layer_sizes = bypass_layer_sizes
    self.bypass_weight_init_stddevs = bypass_weight_init_stddevs
    self.bypass_bias_init_consts = bypass_bias_init_consts
    self.bypass_dropouts = bypass_dropouts
    super(RobustMultitaskRegressor, self).__init__(n_tasks, n_features, logdir,
                                                   **kwargs)

  def build(self, graph, name_scopes, training):
    """Constructs the graph architecture as specified in its config.

    This method creates the following Placeholders:
      mol_features: Molecule descriptor (e.g. fingerprint) tensor with shape
        batch_size x num_features.
    """
    num_features = self.n_features
    placeholder_scope = TensorflowGraph.get_placeholder_scope(
        graph, name_scopes)
    with graph.as_default():
      with placeholder_scope:
        mol_features = tf.placeholder(
            tf.float32, shape=[None, num_features], name='mol_features')

      layer_sizes = self.layer_sizes
      weight_init_stddevs = self.weight_init_stddevs
      bias_init_consts = self.bias_init_consts
      dropouts = self.dropouts

      bypass_layer_sizes = self.bypass_layer_sizes
      bypass_weight_init_stddevs = self.bypass_weight_init_stddevs
      bypass_bias_init_consts = self.bypass_bias_init_consts
      bypass_dropouts = self.bypass_dropouts

      lengths_set = {
          len(layer_sizes),
          len(weight_init_stddevs),
          len(bias_init_consts),
          len(dropouts),
      }
      assert len(lengths_set) == 1, "All layer params must have same length."
      num_layers = lengths_set.pop()
      assert num_layers > 0, "Must have some layers defined."

      bypass_lengths_set = {
          len(bypass_layer_sizes),
          len(bypass_weight_init_stddevs),
          len(bypass_bias_init_consts),
          len(bypass_dropouts),
      }
      assert len(bypass_lengths_set) == 1, (
          "All bypass_layer params" + " must have same length.")
      num_bypass_layers = bypass_lengths_set.pop()

      label_placeholders = self.add_label_placeholders(graph, name_scopes)
      weight_placeholders = self.add_example_weight_placeholders(
          graph, name_scopes)
      if training:
        graph.queue = tf.FIFOQueue(
            capacity=5,
            dtypes=[tf.float32] *
            (len(label_placeholders) + len(weight_placeholders) + 1))
        graph.enqueue = graph.queue.enqueue(
            [mol_features] + label_placeholders + weight_placeholders)
        queue_outputs = graph.queue.dequeue()
        labels = queue_outputs[1:len(label_placeholders) + 1]
        weights = queue_outputs[len(label_placeholders) + 1:]
        prev_layer = queue_outputs[0]
      else:
        labels = label_placeholders
        weights = weight_placeholders
        prev_layer = mol_features

      top_layer = prev_layer
      prev_layer_size = num_features
      for i in range(num_layers):
        # layer has shape [None, layer_sizes[i]]
        print("Adding weights of shape %s" % str(
            [prev_layer_size, layer_sizes[i]]))
        layer = tf.nn.relu(
            model_ops.fully_connected_layer(
                tensor=prev_layer,
                size=layer_sizes[i],
                weight_init=tf.truncated_normal(
                    shape=[prev_layer_size, layer_sizes[i]],
                    stddev=weight_init_stddevs[i]),
                bias_init=tf.constant(
                    value=bias_init_consts[i], shape=[layer_sizes[i]])))
        layer = model_ops.dropout(layer, dropouts[i], training)
        prev_layer = layer
        prev_layer_size = layer_sizes[i]

      output = []
      # top_multitask_layer has shape [None, layer_sizes[-1]]
      top_multitask_layer = prev_layer
      for task in range(self.n_tasks):
        # TODO(rbharath): Might want to make it feasible to have multiple
        # bypass layers.
        # Construct task bypass layer
        prev_bypass_layer = top_layer
        prev_bypass_layer_size = num_features
        for i in range(num_bypass_layers):
          # bypass_layer has shape [None, bypass_layer_sizes[i]]
          print("Adding bypass weights of shape %s" % str(
              [prev_bypass_layer_size, bypass_layer_sizes[i]]))
          bypass_layer = tf.nn.relu(
              model_ops.fully_connected_layer(
                  tensor=prev_bypass_layer,
                  size=bypass_layer_sizes[i],
                  weight_init=tf.truncated_normal(
                      shape=[prev_bypass_layer_size, bypass_layer_sizes[i]],
                      stddev=bypass_weight_init_stddevs[i]),
                  bias_init=tf.constant(
                      value=bypass_bias_init_consts[i],
                      shape=[bypass_layer_sizes[i]])))

          bypass_layer = model_ops.dropout(bypass_layer, bypass_dropouts[i],
                                           training)
          prev_bypass_layer = bypass_layer
          prev_bypass_layer_size = bypass_layer_sizes[i]
        top_bypass_layer = prev_bypass_layer

        if num_bypass_layers > 0:
          # task_layer has shape [None, layer_sizes[-1] + bypass_layer_sizes[-1]]
          task_layer = tf.concat(
              axis=1, values=[top_multitask_layer, top_bypass_layer])
          task_layer_size = layer_sizes[-1] + bypass_layer_sizes[-1]
        else:
          task_layer = top_multitask_layer
          task_layer_size = layer_sizes[-1]
        print("Adding output weights of shape %s" % str([task_layer_size, 1]))
        output.append(
            tf.squeeze(
                model_ops.fully_connected_layer(
                    tensor=task_layer,
                    size=1,
                    weight_init=tf.truncated_normal(
                        shape=[task_layer_size, 1],
                        stddev=weight_init_stddevs[-1]),
                    bias_init=tf.constant(
                        value=bias_init_consts[-1], shape=[1])),
                axis=1))
      return (output, labels, weights)
