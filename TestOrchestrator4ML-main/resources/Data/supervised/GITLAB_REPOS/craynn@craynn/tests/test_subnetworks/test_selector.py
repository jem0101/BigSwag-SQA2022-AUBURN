def test_selector(tf):
  from craynn.subnetworks import with_inputs

  assert with_inputs(0, 1)(
    lambda *xs: [x + 1 for x in xs]
  )(1, 2, 3, 4) == [2, 3, 3, 4]

  assert with_inputs(1, 3)(
    lambda *xs: [x + 1 for x in xs]
  )(10, 20, 30, 40) == [10, 21, 30, 41]

  assert with_inputs(1, 2)(
    lambda *xs: [ x + 1 for x in xs ] + [ sum(xs) ]
  )(10, 20, 30, 40) == [10, 21, 31, 40, 50]

  assert with_inputs(0, 1)(
    lambda *xs: sum(xs)
  )(1, 2, 3, 4) == [3, 3, 4]