import tensorflow as tf

__all__ = [
  'Objective'
]

class Objective(object):
  def __add__(self, other):
    from .common import SumOfObjectives

    if isinstance(other, SumOfObjectives):
      return other.__radd__(self)
    else:
      return SumOfObjectives((self, other))

  def __radd__(self, other):
    from .common import SumOfObjectives

    if isinstance(other, SumOfObjectives):
      return other + self
    else:
      return SumOfObjectives((other, self))

  def __mul__(self, other):
    from .common import ScaledObjective

    if isinstance(other, (int, float)) or tf.is_tensor(other):
      return ScaledObjective(objective=self, coef=other)
    else:
      raise Exception('Multiplication is only allowed for an objective and a constant')

  def __rmul__(self, other):
    return self.__mul__(other)

  def __truediv__(self, other):
    if isinstance(other, (int, float)) or tf.is_tensor(other):
      return self * (1.0 / other)
    else:
      raise Exception('Division is only allowed for an objective and a constant')

  def __rtruediv__(self, other):
    from .common import InvertedObjective
    if isinstance(other, (int, float)) or tf.is_tensor(other):
      return InvertedObjective(objective=self, coef=other)
    else:
      raise Exception('Division is only allowed for an objective and a constant')

  def __call__(self, *args):
    raise NotImplementedError()

