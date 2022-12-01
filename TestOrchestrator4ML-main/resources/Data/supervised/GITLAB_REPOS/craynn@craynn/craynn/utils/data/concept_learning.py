__all__ = [
  'split_groups'
]

def split_groups(data, min_size):
  regrouped = []

  for group in data:
    if len(group) < 2 * min_size:
      regrouped.append(group)
    else:
      n_groups = len(group) // min_size
      group_size = len(group) // n_groups

      for i in range(n_groups - 1):
        regrouped.append(group[i * group_size:(i + 1) * group_size])

      regrouped.append(group[(n_groups - 1) * group_size:])

  return regrouped