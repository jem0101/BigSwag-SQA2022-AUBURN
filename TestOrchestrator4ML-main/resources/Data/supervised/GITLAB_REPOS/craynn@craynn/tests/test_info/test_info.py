def test_info():
  from craynn import network, dense, concat, info

  net = network((None, 8))(
    dense(16), dense(32),
    [dense(12), dense(13)],
    concat(),
    dense(2)
  )

  print()
  print('Input parameters:', net.inputs()[0].parameters())
  print('Input incoming', net.inputs()[0].incoming())
  for l, n in zip(
    net.layers(),
    info.get_number_of_parameters(net.layers())
  ):
    print(l, n)

  print()
  print(net.description(short=True))

  print()
  print(net.description(short=False))


def test_core():
  from craynn import network, dense, concat, info, Layer

  net = network((None, 8))(
    dense(16), dense(32),
    [dense(12), dense(13)],
    concat(),
    dense(2)
  )

  print()
  for node in info.get_network_core(net.outputs()):
    assert isinstance(node, Layer)
    print(node)