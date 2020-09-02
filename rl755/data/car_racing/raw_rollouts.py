"""Datasets that come from raw rollouts."""
import functools
import tensorflow as tf

from rl755.data.car_racing import processing

TFRECORDS_PATTERN = (
    "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/"
    "raw_rollouts/{split}/raw_rollouts.tfrecord*"
)


def _process_observations(observations):
    observations = tf.map_fn(
        functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
        tf.squeeze(observations),
        dtype=tf.uint8,
    )
    # Convert to floats in the range [0, 1]
    observations = tf.cast(observations, tf.float32) / 255.0
    return observations


def parse_fn(x, process_observations):
    """Parses a single tfrecord.

    Args:
        x: a raw tfrecord or something
        process_observations: bool, whether to parse the "observations" value. If we
            are only extracting a few observations from a rollout in further processing,
            then it can be significantly more efficient to only parse the obervations we
            have selected.
    Returns:
        A dict of tf.Tensors.
    """
    features = {
        "observations": tf.io.VarLenFeature(tf.string),
        "actions": tf.io.VarLenFeature(tf.float32),
        "rewards": tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x["rewards"] = tf.squeeze(x["rewards"])
    if process_observations:
        x["observations"] = _process_observations(x["observations"])
    return x


def get_raw_rollouts_ds(process_observations=True, split="train"):
    """Returns a tf.data.Dataset where each item is a raw full rollout.

    Each example is a dict with tf.Tensor items:
      'observations': tf.float32, [rollout_len, 96, 96, 3]
      'actions': tf.float32, [rollout_len, 4]
      'rewards': tf.float32, [rollout_len]

    Getting the i-th item in each of the tensors will provide values for the
    observation on the i-th step, the action taken at the i-th step, and the
    reward corresponding to the transition from the i-th state to the (i+1)-th
    state.

    Args:
        process_observations: bool, whether to parse the "observations" value. See documentation
            for `parse_fn` for more details.
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
    ds = ds.repeat()
    return ds.map(
        functools.partial(parse_fn, process_observations=process_observations),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )


def random_rollout_slices(slice_size, split="train"):
    """Returns a tf.data.Dataset where items are random windows of size `slice_size`.

    See the documentation for `get_raw_rollouts_ds()` for more information about what this
    function returns. The items in the dataset will have the same structure except the
    size of their first dimension will be `slice_size`.

    Args:
        slice_size: a positive integer, the length of each slice
        split: str, "train" or "validation"
    Returns:
        A tf.data.Dataset.
    """

    def map_fn(x):
        x = processing.raw_rollout_vae_32ld(x, slice_size=slice_size)
        x["observations"] = _process_observations(x["observations"])
        return x

    ds = get_raw_rollouts_ds(process_observations=False, split=split)
    return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_observations(obs_per_rollout=100, split="train"):
    """Returns a tf.data.Dataset where each item is a single image.

    Each example is a dict with tf.Tensor items:
      'observations': tf.float32, [96, 96, 3]

    Args:
        obs_per_rollout: a positive float, the number of observations to extract from each
            rollout. Extracting too many from each rollout can lead to images being repeated
            and having many correlated examples in each minibatch. Extracting too few can
            lead to poor performance since each raw rollout is large and takes a while to read
            from disk.
       split: str, "train" or "validation"
    Returns:
        A tf.data.Dataset.
    """

    def random_obs(x):
        rollout_length = tf.shape(x["observations"])[0]
        index = tf.random.uniform([obs_per_rollout], 0, rollout_length, dtype=tf.int32)
        observation = tf.gather(x["observations"], index, axis=0)
        observation = _process_observations(observation)
        return {"observation": observation}

    def set_shape(x):
        return {"observation": tf.reshape(x["observation"], (96, 96, 3))}

    ds = get_raw_rollouts_ds(process_observations=False, split=split)
    ds = ds.map(random_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    ds = ds.map(set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds
