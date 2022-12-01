__all__ = [
  'many_to_one',
  'one_to_one',
  'others',
  'default_mapping',

  'encode_text', 'encode_text_groups'
]

class TextMapping(object):
  def table(self):
    import numpy as np
    return np.array([
      self.translate(chr(i))
      for i in range(256)
    ], dtype='uint8')

  def __len__(self):
    raise NotImplementedError()

  def __or__(self, other):
    return UnionMapping(self, other)

  def __add__(self, other):
    return ConcatMapping(self, other)

  def translate(self, character):
    raise NotImplementedError()

  def __call__(self, text):
    return [
      self.translate(c)
      for c in text
    ]

class ConcatMapping(TextMapping):
  def __init__(self, first : TextMapping, second : TextMapping):
    self._first = first
    self._second = second
    self._offset = len(self._second)

  def __len__(self):
    return len(self._first) + len(self._second)

  def translate(self, character):
    result = self._first.translate(character)
    if result is not None:
      return result + self._offset
    else:
      result = self._second.translate(character)
      return result

class UnionMapping(TextMapping):
  def __init__(self, first : TextMapping, second : TextMapping):
    self._first = first
    self._second = second

  def __len__(self):
    return max(len(self._first), len(self._second))

  def translate(self, character):
    result = self._first.translate(character)
    if result is None:
      result = self._second.translate(character)

    return result

class ManyToOneMapping(TextMapping):
  def __init__(self, characters):
    self._characters = set(characters)
    super(ManyToOneMapping, self).__init__()

  def __len__(self):
    return 1

  def translate(self, character):
    return 0 if character in self._characters else None

class OneToOneMapping(TextMapping):
  def __init__(self, characters):
    self._mapping = dict(zip(characters, range(len(characters))))

  def __len__(self):
    return len(self._mapping)

  def translate(self, character):
    return self._mapping.get(character, None)

class EtcMapping(TextMapping):
  def __init__(self):
    super(EtcMapping, self).__init__()

  def __len__(self):
    return 1

  def translate(self, character):
    return 0

many_to_one = ManyToOneMapping
one_to_one = OneToOneMapping
others = EtcMapping

import string
default_mapping = many_to_one(string.whitespace) + (
  one_to_one(string.ascii_lowercase) | one_to_one(string.ascii_uppercase)
) + others()

def encode_text(mapping):
  import numpy as np
  try:
    table = mapping.table()
  except:
    table = mapping

  def f(s):
    array = np.array([ ord(c) for c in s ], dtype='uint8')
    return table[array]

  return f

def encode_text_groups(mapping, fill=' '):
  import numpy as np

  try:
    table = mapping.table()
  except:
    table = mapping

  def f(group):
    group_len = max([ len(s) for s in group ])
    encoded = np.ones(shape=(len(group), group_len), dtype='uint8') * table[ord(fill)]

    for i, s in enumerate(group):
      array = np.array([ord(c) for c in s], dtype='int64')
      encoded[i, :len(s)] = table[array]

    return encoded

  return f