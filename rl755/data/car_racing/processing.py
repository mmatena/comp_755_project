"""Common processing functions for car racing datasets."""
import tensorflow as tf


def standard_dataset_prep(ds, batch_size, repeat=True, shuffle_buffer_size=1000):
  # TODO(mmatena): Add docs.
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  if repeat:
    ds = ds.repeat()
  ds = ds.batch(batch_size, drop_remainder=True)
  return ds


def slice_example(x, slice_size):
  # TODO(mmatena): Add docs.
  # TODO(mmatena): Handle slice_sizes biggers than the rollout length.
  # Note that we assume that all values have the same first dimension.
  rollout_length = tf.shape(next(iter(x.values())))[0]
  slice_start = tf.random.uniform([], 0, rollout_length - slice_size, dtype=tf.int32)
  x = {k: v[slice_start:slice_start + slice_size] for k, v in x.items()}
  return x
