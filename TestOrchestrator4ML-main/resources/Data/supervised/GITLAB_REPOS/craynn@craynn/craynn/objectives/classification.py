import tensorflow as tf

from .meta import Objective

__all__ = [
  'logit_binary_crossentropy', 'logit_categorical_crossentropy', 'logit_crossentropy',
  'batch_logit_binary_crossentropy', 'batch_logit_categorical_crossentropy',
  'binary_crossentropy', 'categorical_crossentropy',

  'per_class_logit_crossentropy'
]

class logit_binary_crossentropy(Objective):
  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 1, \
      'Predictions for a binary loss must be a 1D-tensor, got %s' % (predictions.shape, )
    losses = target * tf.nn.softplus(-predictions) + (1 - target) * tf.nn.softplus(predictions)

    if weights is not None:
      return tf.reduce_mean(weights * losses)
    else:
      return tf.reduce_mean(losses)


class logit_categorical_crossentropy(Objective):
  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 2, \
      'Predictions for a categorical loss must be a 2D-tensor, got %s' % (predictions.shape, )

    ### seems like reduce_logsumexp can safely handle large values.
    neg_log_softmax = tf.math.reduce_logsumexp(predictions, axis=1)[:, None] - predictions

    losses = tf.reduce_sum(target * neg_log_softmax, axis=1)

    if weights is not None:
      return tf.reduce_mean(weights * losses)
    else:
      return tf.reduce_mean(losses)

class batch_logit_binary_crossentropy(Objective):
  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 2, \
      'Predictions for a batch binary loss must be a 2D-tensor, got %s' % (predictions.shape, )
    losses = target[None, :] * tf.nn.softplus(-predictions) + (1 - target)[None, :] * tf.nn.softplus(predictions)

    if weights is not None:
      return tf.reduce_mean(weights * losses, axis=-1)
    else:
      return tf.reduce_mean(losses, axis=-1)


class batch_logit_categorical_crossentropy(Objective):
  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 3, \
      'Predictions for a batch categorical loss must be a 3D-tensor, got %s' % (predictions.shape, )
    ### seems like reduce_logsumexp can safely handle large values.
    neg_log_softmax = tf.math.reduce_logsumexp(predictions, axis=-1)[:, :, None] - predictions

    losses = tf.reduce_sum(target * neg_log_softmax, axis=-1)

    if weights is not None:
      return tf.reduce_mean(weights * losses, axis=-1)
    else:
      return tf.reduce_mean(losses, axis=-1)


class logit_crossentropy(Objective):
  def __call__(self, target, predictions, weights=None):
    if len(target.shape) == 1:
      return logit_binary_crossentropy()(target, predictions, weights)
    else:
      return logit_categorical_crossentropy()(target, predictions, weights)


class binary_crossentropy(Objective):
  """
    - sum_i ( y_i log(f(x_i) + eps) + (1 - y_i) log(1 - f(x_i) + eps) )
    :param eps: if not None replaces x -> log(x) with x -> log(x + eps) for computational stability.
    """
  def __init__(self, eps=1e-3):
    self.eps = eps

  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 1, 'Predictions for binary loss must be a 1D-tensor.'

    if self.eps is None:
      losses = target * tf.math.log(predictions) + (1 - target) * tf.math.log(1 - predictions)
    else:
      losses = target * tf.math.log(predictions + self.eps) + (1 - target) * tf.math.log(1 - predictions + self.eps)

    if weights is not None:
      return -tf.reduce_mean(weights * losses)
    else:
      return -tf.reduce_mean(losses)


class categorical_crossentropy(Objective):
  """
   - sum_i sum_j y_ij log(f_j(x_i) + eps)
   :param eps: if not None replaces x -> log(x) with x -> log(x + eps) for computational stability.
   """
  def __init__(self, eps=1e-3):
    self.eps = eps

  def __call__(self, target, predictions, weights=None):
    assert len(predictions.shape) == 2, 'Predictions for a categorical loss must be a 2D-tensor.'

    if self.eps is None:
      losses = target * tf.math.log(predictions)
    else:
      losses = target * tf.math.log(predictions + self.eps)

    if weights is not None:
      return -tf.reduce_mean(weights * losses)
    else:
      return -tf.reduce_mean(losses)


class crossentropy(Objective):
  """
   Automatically chooses between binary and categorical cross-entropy based on shape of the target.
   :param eps: if not None replaces x -> log(x) with x -> log(x + eps) for computational stability.
   """
  def __init__(self, eps=1e-3):
    self.eps = eps

  def __call__(self, target, predictions, weights=None):
    if len(target.shape) == 1:
      return binary_crossentropy(eps=self.eps)(target, predictions, weights)
    else:
      return categorical_crossentropy(eps=self.eps)(target, predictions, weights)

def _concat(p_neg, p_pos, keep_priors=True):
  predictions = tf.concat([
    p_neg,
    p_pos
  ], axis=0)

  target = tf.concat([
    tf.zeros_like(p_neg),
    tf.ones_like(p_pos),
  ], axis=0)

  if not keep_priors:
    n_neg = tf.shape(p_neg)[0]
    n_pos = tf.shape(p_pos)[1]

    total = n_neg + n_pos

    w_neg = 0.5 * total / n_neg
    w_pos = 0.5 * total / n_pos

    weights = tf.concat([
      w_neg * tf.ones_like(p_neg),
      w_pos * tf.ones_like(p_pos),
    ])
  else:
    weights = None

  return target, predictions, weights

class per_class_logit_crossentropy(Objective):
  def __init__(self, concat=True, keep_priors=True):
    """
    Similar to cross-entropy but accepts predictions for positive and negative classes separately.
    Useful for e.g. discriminators in GAN.

    :param concat: if predictions should be concatenated internally;
      if True typically results in a slightly faster procedure.
    :param keep_priors: whatever to respect relative sizes of prediction tensors, if False,
      assumes priors for each class 0.5.
    """
    self.keep_priors = keep_priors
    self.concat = concat

  def __call__(self, predictions_negative, predictions_positive):
    if self.concat:
      target, predictions, weights = _concat(predictions_negative, predictions_positive, keep_priors=self.keep_priors)
      return logit_binary_crossentropy()(target, predictions, weights)
    else:
      losses_pos = tf.nn.softplus(-predictions_positive)
      losses_neg = tf.nn.softplus(predictions_negative)

      if self.keep_priors:
        total = tf.cast(tf.shape(losses_pos)[0] + tf.shape(losses_neg)[0], dtype=losses_pos.dtype)
        return (tf.reduce_sum(losses_pos) + tf.reduce_sum(losses_neg)) / total
      else:
        return 0.5 * (tf.reduce_mean(losses_pos) + tf.reduce_mean(losses_neg))
