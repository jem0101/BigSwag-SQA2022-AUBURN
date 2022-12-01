def test_shapes(tf):
  from craynn import layers
  from craynn import network, select, seek, with_inputs

  class TrapLayer(layers.Layer):
    def __init__(self, *incoming, shape, name=None):
      self._out_shape = shape
      self._counter = 0

      super(TrapLayer, self).__init__(*incoming, name=name)

    def get_output_shape_for(self, *incoming_shapes):
      self._counter += 1
      return self._out_shape

    def get_output_for(self, *inputs):
      raise Exception('get_output_for call')

  trap = layers.model_from(TrapLayer)()

  net = network((None, 1, 28, 28), (None, 1, 28, 28))(
    with_inputs[0](
      trap((None, 18)), [
        trap((None, 20), name='trap1'), trap((None, 21))
      ],
    ),
    with_inputs[1](
      trap((None, 17))
    ),
    trap((None, 88)),
    [
      trap((None, 8)),
      trap((None, 19)),
      seek('trap1')(
        trap((10, 11))
      )
    ],
    [trap((None, 19)), trap((None, 19))]
  )

  for _ in range(32):
    for l in net.outputs():
      layers.get_output_shape(l)

  from craynn import viz
  viz.draw_to_file('shapes.png', net.outputs(), inputs=None)

  for layer in net.layers():
    if hasattr(layer, '_counter'):
      assert getattr(layer, '_counter') == 1


def test_shapes_params(tf):
  from craynn import layers, parameters
  from craynn import network

  net = network((None, 3, 32, 32))(
    layers.pconv(32, name='pconv'),
  )

  print(layers.get_output_shape(net.outputs()[0]))
