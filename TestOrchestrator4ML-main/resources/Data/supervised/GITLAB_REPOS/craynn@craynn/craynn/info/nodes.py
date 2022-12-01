import inspect

def get_non_default_arguments(node):
  class_signature = inspect.signature(node.__class__)
