import inspect

def test_correctness():
  from craynn import layers

  for item in dir(layers):
    x = getattr(layers, item)

    if not isinstance(x, type):
      continue

    if not issubclass(x, layers.Layer):
      continue

    get_output_for = getattr(x, 'get_output_for', None)
    get_output_shape_for = getattr(x, 'get_output_shape_for', None)

    assert callable(get_output_for), 'Layer %s does not have `get_output_for` method' % (x, )
    assert callable(get_output_shape_for), 'Layer %s does not have `get_output_shape_for` method' % (x, )

    print(x.__name__)
    print(inspect.signature(get_output_for))
    print(inspect.signature(get_output_shape_for))
    print()





