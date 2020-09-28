"""Miscelaneous utilty functions."""


def divide_chunks(lis, n):
    """Yield successive n-sized from list `lis`."""
    for i in range(0, len(lis), n):
        yield lis[i : i + n]


def evenly_partition(lis, n):
    """Partition the list lis into n lists that are as evenly sized as possible.

    If lis is an integer, then we will assume it is the total number of items
    and return a list containing the sizes of the partitions.

    Args:
        lis: either the list to partition or an integer specificying a number to partition
        n: a positive integer specifying the number of partitions
    Returns:
        A list containing the partitions, each of which will either be the list of items
        in the partition or the size of the partition. That behavior depends on the type
        of `lis`. The returned list will always have length `n`.
    """
    if isinstance(lis, int):
        # TODO(mmatena): This could be done in another functions without generating lists.
        return [len(p) for p in evenly_partition(list(range(lis)), n)]

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


def sharded_filename(filename, shard_index, num_shards):
    # TODO(mmatena): Support cases with 10**6 or more shards.
    return f"{filename}-{shard_index:05d}-of-{num_shards:05d}"
