"""Datasets that come from raw rollouts."""
import functools
import tensorflow as tf

_TFRECORDS_PATTERN = "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/" \
                     "raw_rollouts/raw_rollouts.tfrecord*"


def get_raw_rollouts_ds(shuffle_files=True):
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
  files = tf.io.matching_files(_TFRECORDS_PATTERN)
  if shuffle_files:
    files = tf.random.shuffle(files)

  def parse_fn(x):
    features = {
        'observations': tf.io.VarLenFeature(tf.string),
        'actions': tf.io.VarLenFeature(tf.float32),
        'rewards': tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x['rewards'] = tf.squeeze(x['rewards'])
    x['observations'] = tf.map_fn(functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
                                  tf.squeeze(x['observations']), dtype=tf.uint8)
    return x

  ds = tf.data.TFRecordDataset(files)
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_slices(slice_size, shuffle_files=True):
  # TODO(mmatena): Add docs.
  # TODO(mmatena): Handle slice_sizes biggers than the rollout length.

  def slice_example(x):
    # Note that we assume that 'observations', 'rewards', and 'actions' are
    # all the same length.
    rollout_length = tf.shape(x['observations'])[0]
    slice_start = tf.random.uniform([], 0, rollout_length - slice_size, dtype=tf.int32)
    x = {k: v[slice_start:slice_start + slice_size] for k, v in x.items()}
    # # Explicitly set the shape because I think it makes stuff nicer later on.
    # x = {k: tf.reshape(x, tf.concat([[slice_size], tf.shape(v)[1:]], axis=0))
    #      for k, v in x.items()}
    return x

  ds = get_raw_rollouts_ds(shuffle_files=shuffle_files)
  return ds.map(slice_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_observations(shuffle_files=True):
  # TODO(mmatena): Add docs.

  def random_obs(x):
    rollout_length = tf.shape(x['observations'])[0]
    index = tf.random.uniform([], 0, rollout_length, dtype=tf.int32)
    return {"observation": x['observations'][index]}

  ds = get_raw_rollouts_ds(shuffle_files=shuffle_files)
  return ds.map(random_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
