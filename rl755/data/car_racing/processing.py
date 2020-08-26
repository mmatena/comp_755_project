"""Common processing functions for car racing datasets."""
import tensorflow as tf


def standard_dataset_prep(ds, batch_size, repeat=True, shuffle_buffer_size=1000):
    """Common dataset preprocessing for training including batching and shuffling."""
    # TODO(mmatena): Add docs.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def slice_example(x, slice_size):
    """Returns a random slice along the first dimension from tensors in example `x`.

    We assume that all values in `x` have an identical first dimension.

    Args:
        x: a dict of tf.Tensors
        slice_size: a positive integer
    Returns:
        A dict of of tf.Tensors.
    """
    # TODO(mmatena): Handle slice_sizes biggers than the rollout length.
    rollout_length = tf.shape(next(iter(x.values())))[0]
    slice_start = tf.random.uniform([], 0, rollout_length - slice_size, dtype=tf.int32)
    x = {k: v[slice_start : slice_start + slice_size] for k, v in x.items()}
    return x
