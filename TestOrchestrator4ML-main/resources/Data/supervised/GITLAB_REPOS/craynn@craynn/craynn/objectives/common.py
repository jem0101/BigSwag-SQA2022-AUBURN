import tensorflow as tf
from .meta import Objective

__all__ = [
  'identity',
  'SumOfObjectives',
  'ScaledObjective',
  'InvertedObjective',

  'sum'
]

class identity(Objective):
  def __init__(self, always_return_tuple=False):
    self.always_return_tuple = always_return_tuple

  def __call__(self, *args):
    if self.always_return_tuple:
      return args
    else:
      if len(args) == 1:
        return args[0]
      else:
        return args

def get_number_of_arguments(f):
  import inspect
  signature = inspect.signature(f)
  assert not any(
    p.kind == inspect.Parameter.VAR_KEYWORD or p.kind == inspect.Parameter.KEYWORD_ONLY
    for p in signature.parameters
  ), 'variable keyword or keyword-only arguments are not allowed in objectives'

  if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in signature.parameters):
    return slice(None, None)

  return slice(None, len(signature.parameters))

def sum(*objectives):
  return SumOfObjectives(objectives)

class SumOfObjectives(Objective):
  def __init__(self, objectives, number_of_arguments=None):
    self.objectives = tuple(objectives)

    if number_of_arguments is None:
      self.number_of_arguments = tuple(
        get_number_of_arguments(objective)
        for objective in objectives
      )
    else:
      self.num_of_arguments = number_of_arguments

  def __add__(self, other):
    if isinstance(other, SumOfObjectives):
      return SumOfObjectives(
        self.objectives + other.objectives,
        self.number_of_arguments + other.number_of_arguments
      )
    else:
      return SumOfObjectives(
        self.objectives + (other, ),
        self.number_of_arguments + (get_number_of_arguments(other), )
      )

  def __radd__(self, other):
    if isinstance(other, SumOfObjectives):
      ### this code should never be reached
      return SumOfObjectives(
        other.objectives + self.objectives,
        other.number_of_arguments + self.number_of_arguments
      )
    else:
      return SumOfObjectives(
        (other, ) + self.objectives,
        (get_number_of_arguments(other), ) + self.number_of_arguments
      )

  def __call__(self, *args):
    return tuple(
      objective(*args[selector])
      for objective, selector in zip(self.objectives, self.number_of_arguments)
    )


class ScaledObjective(Objective):
  def __init__(self, objective, coef):
    self.objective = objective
    self.coef = tf.convert_to_tensor(coef, dtype=tf.float32)

  def __call__(self, *args):
    return self.coef * self.objective(*args)

  def __mul__(self, other):
    if isinstance(other, (int, float)):
      return ScaledObjective(
        objective=self.objective,
        coef=other * self.coef
      )
    else:
      return super(ScaledObjective, self).__mul__(other)

class InvertedObjective(Objective):
  def __init__(self, objective, coef):
    self.objective = objective
    self.coef = tf.convert_to_tensor(coef)

  def __call__(self, *args):
    return self.coef / self.objective(*args)