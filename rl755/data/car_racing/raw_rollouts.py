"""Datasets that come from raw rollouts."""
import functools
import tensorflow as tf

_TFRECORDS_PATTERN = "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/" \
                     "raw_rollouts/raw_rollouts.tfrecord*"


def get_raw_rollouts_ds(shuffle_files=True):
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

    observations = tf.squeeze(x['observations'])
    observations = tf.map_fn(functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
                             observations, dtype=tf.uint8)
    # observations = tf.unstack(observations)
    # observations = [tf.io.parse_tensor(obs, out_type=tf.uint8) for obs in observations]
    x['observations'] = tf.stack(observations)

    return x

  ds = tf.data.TFRecordDataset(files)
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


for x in get_raw_rollouts_ds():
  print(x)
  break
