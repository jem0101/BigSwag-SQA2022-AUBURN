import tensorflow as tf
from craygraph import derive

from .meta import Layer, InputLayer
from .meta import model_from

from ..parameters import default_input_init

__all__ = [
  'const_input',
  
  'FunctionLayer', 'custom', 'nonlinearity',
  'ConcatLayer', 'concat',
  'FlattenLayer', 'flatten',
  'ReshapeLayer', 'reshape',
  'TransposeLayer', 'transpose',
  'ExpandLayer', 'expand',
  'RepeatLayer', 'repeat',

  'ElementwiseLayer', 'ElementwiseSumLayer', 'ElementwiseMeanLayer',
  'ElementwiseMaxLayer', 'ElementwiseMinLayer',

  'elementwise', 'elementwise_sum', 'elementwise_mean',
  'elementwise_max', 'elementwise_min',

  'SquareDifference', 'square_difference',

  'BroadcastConcatLayer', 'broadcast_concat',

  'GeneralPoolLayer', 'GeneralMaxPoolLayer', 'GeneralMeanPoolLayer',
  'general_max_pool', 'general_mean_pool',

  'BatchPool', 'MaxBatchPool', 'MeanBatchPool',
  'max_batch_pool', 'mean_batch_pool'
]

class ConstInput(InputLayer):
  def __init__(self, const, name=None):
    self.value = tf.constant(const)
    super(ConstInput, self).__init__(shape=const.shape, name=name)

  def get_output_for(self):
    return self.value

const_input = model_from(ConstInput)()


_default_shape_f = lambda *input_shapes: input_shapes[0]

class CustomLayer(Layer):
  def __init__(self, f, shape_f=_default_shape_f, *incoming, name=None):
    super(CustomLayer, self).__init__(*incoming, name=name)
    self.f = f
    self.shape_f = shape_f

  def get_output_for(self, *inputs):
    return self.f(*inputs)

  def get_output_shape_for(self, *input_shapes):
    return self.shape_f(*input_shapes)

custom_layer = model_from(CustomLayer)()


class FunctionLayer(Layer):
  def __init__(self, f, *incoming, name=None):
    if name is None:
      name = f.__name__

    super(FunctionLayer, self).__init__(*incoming, name=name)

    self.f = f

  def get_output_for(self, *incoming):
    return self.f(*incoming)

  def get_output_shape_for(self, *input_shapes):
    return input_shapes[0]

custom = model_from(FunctionLayer)()
nonlinearity = custom


class ConcatLayer(Layer):
  def __init__(self, *incoming, axis=-1, name=None):
    assert len(incoming) > 0
    self.axis = axis

    super(ConcatLayer, self).__init__(*incoming, name=name)

  def get_output_for(self, *inputs):
    return tf.concat(values=inputs, axis=self.axis, name=str(self) + '_concat')

  def get_output_shape_for(self, *input_shapes):
    first = input_shapes[0]
    axis = (self.axis + len(first)) % len(first)

    return tuple(
      first[i] if i != axis else sum(s[i] for s in input_shapes)
      for i in range(len(first))
    )

concat = model_from(ConcatLayer)()


from functools import reduce as _reduce

class FlattenLayer(Layer):
  def __init__(self, incoming, outdim=2, name=None):
    self.outdim = outdim

    super(FlattenLayer, self).__init__(incoming, name=name)

  def get_output_for(self, incoming):
    in_shape = tf.shape(incoming)

    out_shape = tf.stack([
      in_shape[i] for i in range(self.outdim - 1)
    ] + [
      tf.reduce_prod(in_shape[self.outdim - 1:])
    ])

    return tf.reshape(incoming, out_shape)

  def get_output_shape_for(self, input_shapes):
    return input_shapes[:self.outdim - 1] + (
      _reduce(
        lambda a, b: a * b if a is not None and b is not None else None,
        input_shapes[self.outdim - 1:],
        1
      ),
    )

flatten = model_from(FlattenLayer)()


class ReshapeLayer(Layer):
  def __init__(self, incoming, new_shape, name=None):

    assert len([dim for dim in new_shape if (dim is None or dim < 0)]) < 2, 'ambiguous new shape'

    self.new_shape = tuple(
      (-1 if s is None else s)
      for s in new_shape
    )

    super(ReshapeLayer, self).__init__(incoming, name=name)

  def get_output_for(self, incoming):
    return tf.reshape(incoming, self.new_shape, name=str(self) + '_reshape')

  def get_output_shape_for(self, input_shape):
    import numpy as np

    if -1 in self.new_shape:
      if all(dim is not None for dim in input_shape):
        total = np.prod(input_shape, dtype='int64')
        known_dims = np.prod([ dim for dim in self.new_shape if dim is not None], dtype='int64')
        assert total % known_dims == 0, 'can not broadcast %s into %s' % (input_shape, self.new_shape)
        inferred = total // known_dims

        return tuple(dim if dim is not None else inferred for dim in self.new_shape)

      else:
        return tuple(dim if dim >= 0 else None for dim in self.new_shape)

    else:
      return self.new_shape

reshape = model_from(ReshapeLayer)()


class TransposeLayer(Layer):
  def __init__(self, incoming, perm, name=None):
    super(TransposeLayer, self).__init__(incoming, name=name)
    self.perm = perm

  def get_output_shape_for(self, input_shape):
    return tuple(input_shape[i] for i in self.perm)

  def get_output_for(self, input):
    return tf.transpose(input, perm=self.perm)

transpose = model_from(TransposeLayer)()


class ExpandLayer(Layer):
  def __init__(self, incoming, item, name=None):
    assert all(
      it == slice(None, None, None) or it is None
      for it in item
    )
    self.item = item

    super(ExpandLayer, self).__init__(incoming, name=name)

  def get_output_shape_for(self, input_shape):
    result = list()
    input_shape_indx = 0

    for i, it in enumerate(self.item):
      if it is None:
        result.append(1)
      else:
        result.append(input_shape[input_shape_indx])
        input_shape_indx += 1

    return tuple(result)

  def get_output_for(self, input):
    return input[self.item]

_expand = model_from(ExpandLayer)()

class ExpandConstructor(object):
  def __call__(self, item, name=None):
    return _expand(item, name=name)

  def __getitem__(self, item):
    return _expand(item)

expand = ExpandConstructor()


class RepeatLayer(Layer):
  def __init__(self, incoming, repeats, axis=1, name=None):
    self.repeats = repeats
    self.axis = axis

    super(RepeatLayer, self).__init__(incoming, name=name)

  def get_output_for(self, input):
    expanded = tf.expand_dims(input, axis=self.axis)
    return tf.repeat(expanded, [self.repeats], axis=self.axis)

  def get_output_shape_for(self, input_shape):
    return input_shape[:self.axis] + (self.repeats, ) + input_shape[self.axis:]

repeat = model_from(RepeatLayer)()


class BroadcastConcatLayer(Layer):
  def __init__(self, *incoming, axis=-1, name=None):
    from .meta import get_output_shape

    assert len(incoming) > 0
    self.axis = axis
    self.ndim = len(get_output_shape(incoming[0]))

    super(BroadcastConcatLayer, self).__init__(*incoming, name=name)

  def get_output_for(self, *inputs):
    from ..utils import normalize_axis
    if len(inputs) == 1:
      return inputs[0]

    original, rest = inputs[0], inputs[1:]
    shape = tf.shape(original)

    normalized_axis = normalize_axis(original, self.axis)

    broadcasted = [original]
    for x in rest:
      x_shape = tf.shape(x)
      new_shape = tuple(
        (x_shape[axis] if axis == normalized_axis else shape[axis])
        for axis in range(self.ndim)
      )
      broadcasted.append(tf.broadcast_to(x, shape=new_shape))

    return tf.concat(broadcasted, axis=normalized_axis)

  def get_output_shape_for(self, *input_shapes):
    from ..utils import normalize_axis, gsum
    if len(input_shapes) == 1:
      return input_shapes

    original, rest = input_shapes[0], input_shapes[1:]
    normalized_axis = normalize_axis(original, self.axis)

    return tuple(
      gsum(shape[i] for shape in input_shapes)if i == normalized_axis else original[i]
      for i in range(len(original))
    )

broadcast_concat = model_from(BroadcastConcatLayer)()


class ElementwiseLayer(Layer):
  def __init__(self, *incoming, op, name=None):
    super(ElementwiseLayer, self).__init__(*incoming, name=name)
    self.op = op

  def get_output_for(self, *inputs):
    return self.op(*inputs)

  def get_output_shape_for(self, *input_shapes):
    for shape in input_shapes[1:]:
      if tuple(shape) != tuple(input_shapes[0]):
        raise ValueError('An elementwise operation requires all input shapes to be the same!')

    return input_shapes[0]

elementwise = model_from(ElementwiseLayer)()


ElementwiseSumLayer = derive('ElementwiseSumLayer').based_on(ElementwiseLayer).with_fixed(
  op=lambda *inputs: sum(inputs)
)
elementwise_sum = model_from(ElementwiseSumLayer)()

ElementwiseMeanLayer = derive('ElementwiseMeanLayer').based_on(ElementwiseLayer).with_fixed(
  op=lambda *inputs: sum(inputs) / len(inputs)
)
elementwise_mean = model_from(ElementwiseMeanLayer)()

from functools import reduce

ElementwiseMaxLayer = derive('ElementwiseMaxLayer').based_on(ElementwiseLayer).with_fixed(
  op=lambda *inputs: reduce(tf.maximum, inputs)
)
elementwise_max = model_from(ElementwiseMaxLayer)()

ElementwiseMinLayer = derive('ElementwiseMinLayer').based_on(ElementwiseLayer).with_fixed(
  op=lambda *inputs: reduce(tf.minimum, inputs)
)
elementwise_min = model_from(ElementwiseMinLayer)()


class SquareDifference(Layer):
  def __init__(self, incoming1, incoming2, axes=None, name=None):
    from .meta import get_output_shape

    if axes is None:
      shape = get_output_shape(incoming1)
      self.axes = list(range(len(shape)))[1:]
    else:
      self.axes = axes

    super(SquareDifference, self).__init__(incoming1, incoming2, name=name)

  def get_output_for(self, input1, input2):
    return tf.reduce_mean(
      (input1 - input2) ** 2,
      axis=self.axes
    )

  def get_output_shape_for(self, shape1, shape2):
    return tuple(
      s for i, s in enumerate(shape1)
      if i not in self.axes
    )

square_difference = model_from(SquareDifference, incoming_args=['incoming1', 'incoming2'])()


class GeneralPoolLayer(Layer):
  def __init__(self, incoming, op, axis=(-1, ), name=None):
    super(GeneralPoolLayer, self).__init__(incoming, name=name)
    self.op = op
    if isinstance(axis, int):
      axis = (axis, )

    self.axis = axis

  def get_output_for(self, input):
    return self.op(input, axis=self.axis, keepdims=False)

  def get_output_shape_for(self, input_shape):
    normalized_axis = tuple(
      (ax + len(input_shape)) % len(input_shape)
      for ax in self.axis
    )

    return tuple(
      dim
      for axis, dim in enumerate(input_shape)
      if axis not in normalized_axis
    )

GeneralMaxPoolLayer = derive('GeneralMaxPoolLayer').based_on(GeneralPoolLayer).with_fixed(op=tf.reduce_max)
GeneralMeanPoolLayer = derive('GeneralMeanPoolLayer').based_on(GeneralPoolLayer).with_fixed(op=tf.reduce_mean)

general_max_pool = model_from(GeneralMaxPoolLayer)()
general_mean_pool = model_from(GeneralMeanPoolLayer)()


class BatchPool(Layer):
  def __init__(self, incoming, op, axes=(0, )):
    super(BatchPool, self).__init__(incoming)
    self.axes = axes
    self.op = op

  def get_output_shape_for(self, input_shape):
    return input_shape

  def get_output_for(self, input):
    averaged = self.op(input, axis=self.axes, keepdims=True)
    return tf.broadcast_to(averaged, shape=tf.shape(input))

MeanBatchPool = derive('MeanBatchPool').based_on(BatchPool).with_fixed(op=tf.reduce_mean)
MaxBatchPool = derive('MaxBatchPool').based_on(BatchPool).with_fixed(op=tf.reduce_max)

mean_batch_pool = model_from(MeanBatchPool)()
max_batch_pool = model_from(MaxBatchPool)()
