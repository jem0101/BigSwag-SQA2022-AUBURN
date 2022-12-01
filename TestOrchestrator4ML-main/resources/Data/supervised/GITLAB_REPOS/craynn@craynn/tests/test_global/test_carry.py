from types import ModuleType

def walk_module(module):
  root = module.__name__

  for k in dir(module):
    item = getattr(module, k)

    if type(item) == ModuleType:
      if not item.__name__.startswith(root):
        for i in walk_module(item):
          yield i
    else:
      yield '%s.%s' % (root, k), item

def test_leftover_carries():
  from craygraph import CarryingExpression
  import craynn

  for path, item in walk_module(craynn):
    assert type(item) != CarryingExpression, '%s is an incomplete carring expression' % (path, )