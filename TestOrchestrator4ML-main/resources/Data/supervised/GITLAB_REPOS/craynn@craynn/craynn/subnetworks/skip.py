from ..layers import Layer
from .meta import achain

__all__ = [
  'skip_connection'
]

def skip_connection(*body):
  def op(*incoming):
    result = achain(*body)(*incoming)
    if isinstance(result, Layer):
      return tuple(incoming) + (result, )
    else:
      return tuple(incoming) + tuple(result)

  return op