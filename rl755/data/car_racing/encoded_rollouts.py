"""Datasets that come from rollouts where some encoding has occured.

Most of the encoding was done by the VAE at
`rl755.models.car_racing.saved_models.raw_rollout_vae_32ld`.
"""
import functools

import tensorflow as tf

from rl755.data.car_racing import processing

# mmatena: I forgot the tfrecord suffix when writing the files.
TFRECORDS_PATTERN = (
    "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/"
    "encoded_rollouts/{split}/encoded_rollouts*"
)


def parse_fn(x):
    """Parses a single tfrecord."""
    features = {
        "observations": tf.io.VarLenFeature(tf.float32),
        "observation_std_devs": tf.io.VarLenFeature(tf.float32),
        "actions": tf.io.VarLenFeature(tf.float32),
        "rewards": tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x["rewards"] = tf.squeeze(x["rewards"])
    return x


def get_rollouts_ds(split="train"):
    """Returns a tf.data.Dataset where each item is a encoded full rollout.

    Each example is a dict with tf.Tensor items:
      'observations': tf.float32, [rollout_len, 32]
      'observations_std_dev': tf.float32, [rollout_len, 32]
      'actions': tf.float32, [rollout_len, 4]
      'rewards': tf.float32, [rollout_len]

    Since the examples were encoded with a VAE, the 'observations' key corresponds
    to the mean of the posterior and the 'observations_std_dev' coresponds to the
    standard deviation of the posterior.

    Getting the i-th item in each of the tensors will provide values for the
    observation on the i-th step, the action taken at the i-th step, and the
    reward corresponding to the transition from the i-th state to the (i+1)-th
    state.

    Args:
        split: str, "train" or "validation"
    Returns:
        A tf.data.Dataset.
    """
    files = tf.io.matching_files(TFRECORDS_PATTERN.format(split=split))

    files = tf.data.Dataset.from_tensor_slices(files)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_slices(slice_size, split="train"):
    """Returns a tf.data.Dataset where items are random windows of size `slice_size`.

    See the documentation for `get_rollouts_ds()` for more information about what this
    function returns. The items in the dataset will have the same structure except the
    size of their first dimension will be `slice_size`.

    Args:
        slice_size: a positive integer, the length of each slice
        split: str, "train" or "validation"
    Returns:
        A tf.data.Dataset.
    """
    ds = get_rollouts_ds(split=split)
    return ds.map(
        functools.partial(processing.slice_example, slice_size=slice_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
