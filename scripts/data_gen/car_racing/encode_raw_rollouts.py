"""Train a VAE on random images collected from the rollouts."""
import os
import time

from absl import app
from absl import flags
import tensorflow as tf

from rl755.common import misc
from rl755.common import structs
from rl755.data.car_racing import raw_rollouts
from rl755.models.car_racing import saved_models

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_shards', None,
                     'The number of shards to use in the written dataset. '
                     'A good goal is to strive for shards of size 100-200 MB.',
                     lower_bound=1)
flags.DEFINE_string('out_dir', None, 'The directory to write the tfrecords to.')
flags.DEFINE_string('out_name', None, 'Prefix to give the generated tfrecord files.')

flags.DEFINE_string('model', "raw_rollout_vae_32ld",
                    'Name of model in `saved_models` to use.')

flags.mark_flag_as_required('num_shards')
flags.mark_flag_as_required('out_dir')
flags.mark_flag_as_required('out_name')


def load_model(model_name):
  return getattr(saved_models, model_name)()


@tf.function
def encode(model, x):
  return model.encode(x)


def run_shard(model, ds, mirrored_strategy, shard_index, num_shards):
  ds = ds.shard(num_shards=num_shards, index=shard_index)
  ds = mirrored_strategy.experimental_distribute_dataset(ds)

  filepath = os.path.join(FLAGS.out_dir,
                          misc.sharded_filename(FLAGS.out_name,
                                                shard_index=shard_index,
                                                num_shards=num_shards))
  with tf.io.TFRecordWriter(filepath) as file_writer:
    for x in ds:
      # TODO(mmatena): This is tailored to VAEs. Handle non-VAE encoders.
      raw_observations = tf.reshape(x['observations'], (-1, 96, 96, 3))
      observations = encode(model, raw_observations)
      file_writer.write(
          structs.latent_image_rollout_to_tfrecord(
              obs_latents=observations.mean(),
              actions=x['actions'],
              rewards=x['rewards'],
              obs_std_devs=observations.stddev()))


def run(ds, num_shards):
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = load_model(FLAGS.model)
    for i in range(num_shards):
      run_shard(model, ds=ds,
                mirrored_strategy=mirrored_strategy,
                shard_index=i, num_shards=num_shards)


def main(_):
  ds = raw_rollouts.get_raw_rollouts_ds(process_observations=True)

  # TODO: REMOVE, THIS IS JUST FOR INITIAL TESTING!!!!!!!!!!!!!!!!!!!!!
  ds = ds.take(128)

  start = time.time()

  run(ds, FLAGS.num_shards)

  end = time.time()
  print("Took", end - start, "seconds to run.")


if __name__ == '__main__':
  app.run(main)
