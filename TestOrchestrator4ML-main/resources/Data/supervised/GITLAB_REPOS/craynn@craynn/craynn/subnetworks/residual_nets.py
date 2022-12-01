from .meta import achain
from ..layers import *

def residual_connection(*body, merge_op=elementwise_mean()):
  def constructor(incoming):
    origin = incoming
    net = achain(*body)(incoming)
    return merge_op(origin, net)

  return constructor
