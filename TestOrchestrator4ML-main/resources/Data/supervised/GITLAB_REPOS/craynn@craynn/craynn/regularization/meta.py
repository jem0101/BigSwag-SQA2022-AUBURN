__all__ = [
  'apply_regularization'
]

def apply_regularization(layer, reg, **properties):
  from ..layers import get_output
  from ..parameters import get_all_parameters

  params = get_all_parameters(layer, **properties)
  values = get_output(params)

  regularizers = [
    reg(param, value)
    for param, value in zip(params, values)
  ]

  return sum(regularizers)
