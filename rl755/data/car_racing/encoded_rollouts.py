"""Datasets that come from rollouts where some encoding has occured.

Most of the encoding was done by the VAE at
`rl755.models.car_racing.saved_models.raw_rollout_vae_32ld`.

# TODO(mmatena): A lot of the code is very similar to the code in `raw_rollouts`.
# Refactor to get rid of this code duplication.
"""
import functools

import tensorflow as tf

from rl755.data.car_racing import processing

# mmatena: I forgot the tfrecord suffix when writing the files.
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
  """Returns a tf.data.Dataset where each item is a encoded full rollout.

  Each example is a dict with items:
    'observations': tf.uint8, (rollout_len, 32)
    'observations_std_dev': tf.uint8, (rollout_len, 32)
    'actions': tf.float32, (rollout_len, 4)
    'rewards': tf.float32, (rollout_len)

  Since the examples were encoded with a VAE, the 'observations' key corresponds
  to the mean of the posterior and the 'observations_std_dev' coresponds to the
  standard deviation of the posterior.

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
  return ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_slices(slice_size):
  # TODO(mmatena): Add docs.
  ds = get_rollouts_ds()
  return ds.map(functools.partial(processing.slice_example, slice_size=slice_size),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def random_rollout_observations(obs_per_rollout=100):
  # TODO(mmatena): Add docs. Mention that obs_per_rollout is because ...
  # TODO(mmatena): Mention that the we are taking the mean of the posterior as the latent.
  def random_obs(x):
    rollout_length = tf.shape(x['observations'])[0]
    index = tf.random.uniform([obs_per_rollout], 0, rollout_length, dtype=tf.int32)
    observation = tf.gather(x['observations'], index, axis=0)
    return {"observation": observation}

  def set_shape(x):
    return {"observation": tf.reshape(x['observation'], (96, 96, 3))}

  ds = get_rollouts_ds()
  ds = ds.map(random_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.map(set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)



for x in random_rollout_slices(slice_size=13):
  print(x)
  break

print("\n\n\n\n\n")

for x in random_rollout_observations():
  print(x)
  break
