"""Miscelaneous utilty functions."""


def divide_chunks(lis, n):
  """Yield successive n-sized from list lis."""
  for i in range(0, len(lis), n):
    yield lis[i:i + n]


def evenly_partition(lis, n):
  """Partition the list lis into n lists that are as evenly sized as possible.

  If lis is an integer, then we will assume it is the total number of items
  and return a list containing the sizes of the partitions.
  """
  if isinstance(lis, int):
    # TODO(mmatena): This could be done in another functions without generating lists.
    return [len(p) for p in evenly_partition(range(lis), n)]

  base_partition_size = len(lis) // n
  partitions = list(divide_chunks(lis, base_partition_size))

  if not (len(lis) % n):
    # This should never fail if the code is correct. However, let's put it here
    # just to prevent bugs.
    assert len(partitions) == n
    return partitions

  partitions, excess_items = partitions[:-1], partitions[-1]
  for partition, item in zip(partitions, excess_items):
    partition.append(item)

  # This should never fail if the code is correct. However, let's put it here
  # just to prevent bugs.
  assert len(partitions) == n

  return partitions