"""Datasets that come from raw rollouts."""
import functools
import tensorflow as tf

TFRECORDS_PATTERN = "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/" \
                    "raw_rollouts/raw_rollouts.tfrecord*"


def _process_observations(observations):
  observations = tf.map_fn(functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
                           tf.squeeze(observations), dtype=tf.uint8)
  # Convert to floats in the range [0, 1]
  observations = tf.cast(observations, tf.float32) / 255.0
  return observations


def parse_fn(x, process_observations):
    features = {
        'observations': tf.io.VarLenFeature(tf.string),
        'actions': tf.io.VarLenFeature(tf.float32),
        'rewards': tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x['rewards'] = tf.squeeze(x['rewards'])
    if process_observations:
      x['observations'] = _process_observations(x['observations'])
    return x


def get_raw_rollouts_ds(process_observations=True):
  """Returns a tf.data.Dataset where each item is a raw full rollout.

  Each example is a dict with items:
    'observations': tf.uint8, (rollout_len, 96, 96, 3)
    'actions': tf.float32, (rollout_len, 4)
    'rewards': tf.float32, (rollout_len)

  Getting the i-th item in each of the tensors will provide values for the
  observation on the i-th step, the action taken at the i-th step, and the
  reward corresponding to the transition from the i-th state to the (i+1)-th
  state.
  """
  files = tf.io.matching_files(TFRECORDS_PATTERN)

  files = tf.data.Dataset.from_tensor_slices(files)
  ds = files.interleave(tf.data.TFRecordDataset,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds.map(functools.partial(parse_fn, process_observations=process_observations),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_slices(slice_size):
  # TODO(mmatena): Add docs.
  # TODO(mmatena): Handle slice_sizes biggers than the rollout length.

  def slice_example(x):
    # Note that we assume that 'observations', 'rewards', and 'actions' are
    # all the same length.
    rollout_length = tf.shape(x['observations'])[0]
    slice_start = tf.random.uniform([], 0, rollout_length - slice_size, dtype=tf.int32)
    x = {k: v[slice_start:slice_start + slice_size] for k, v in x.items()}
    x['observations'] = _process_observations(x['observations'])
    return x

  ds = get_raw_rollouts_ds(process_observations=False)
  return ds.map(slice_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_observations(obs_per_rollout=100, ):
  # TODO(mmatena): Add docs. Mention that obs_per_rollout is because ...
  def random_obs(x):
    rollout_length = tf.shape(x['observations'])[0]
    index = tf.random.uniform([obs_per_rollout], 0, rollout_length, dtype=tf.int32)
    observation = tf.gather(x['observations'], index, axis=0)
    observation = _process_observations(observation)
    return {"observation": observation}

  def set_shape(x):
    return {"observation": tf.reshape(x['observation'], (96, 96, 3))}

  ds = get_raw_rollouts_ds(process_observations=False)
  ds = ds.map(random_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return ds


def standard_dataset_prep(ds, batch_size, repeat=True, shuffle_buffer_size=1000):
  # TODO(mmatena): Add docs.
  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  if repeat:
    ds = ds.repeat()
  ds = ds.batch(batch_size, drop_remainder=True)
  return ds
