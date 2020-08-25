"""Datasets that come from rollouts where some encoding has occured."""
import tensorflow as tf

TFRECORDS_PATTERN = "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/" \
                    "encoded_rollouts/encoded_rollouts*"


def parse_fn(x):
    features = {
        'observations': tf.io.VarLenFeature(tf.float32),
        'observation_std_devs': tf.io.VarLenFeature(tf.float32),
        'actions': tf.io.VarLenFeature(tf.float32),
        'rewards': tf.io.VarLenFeature(tf.float32),
    }
    _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
    x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
    x['rewards'] = tf.squeeze(x['rewards'])
    return x


def get_rollouts_ds():
  # TODO(mmatena): Add docs.
  files = tf.io.matching_files(TFRECORDS_PATTERN)

  files = tf.data.Dataset.from_tensor_slices(files)
  ds = files.interleave(tf.data.TFRecordDataset,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_slices(slice_size):
  # TODO(mmatena): Add docs.
  # TODO(mmatena): Handle slice_sizes biggers than the rollout length.
  pass


def random_rollout_observations(obs_per_rollout=100, ):
  # TODO(mmatena): Add docs. Mention that obs_per_rollout is because ...
  pass




import itertools
for x in itertools.islice(get_rollouts_ds(), 1):
  print(x)
