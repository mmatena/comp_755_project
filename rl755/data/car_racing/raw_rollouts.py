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
    'rewards': tf.float32, (rollout_len, 1)

  Getting the i-th item in each of the tensors will provide values for the
  observation on the i-th step, the action taken at the i-th step, and the
  reward corresponding to the transition from the i-th state to the (i+1)-th
  state.
  """
  files = tf.io.matching_files(_TFRECORDS_PATTERN)
  if shuffle_files:
    files = tf.random.shuffle(files)

  @tf.function
  def parse_fn(x):
    features = {
        'observations': tf.io.VarLenFeature(tf.string),
        'actions': tf.io.VarLenFeature(tf.float32),
        'rewards': tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x['observations'] = tf.map_fn(functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
                                  tf.squeeze(x['observations']), dtype=tf.uint8)
    return x

  ds = tf.data.TFRecordDataset(files)
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
