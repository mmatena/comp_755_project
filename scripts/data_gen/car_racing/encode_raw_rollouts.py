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
flags.DEFINE_integer('batch_size', 8,
                     'The number of rollouts in each batch.',
                     lower_bound=1)

flags.mark_flag_as_required('num_shards')
flags.mark_flag_as_required('out_dir')
flags.mark_flag_as_required('out_name')


def load_model(model_name):
  return getattr(saved_models, model_name)()


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
      raw_observations = tf.reshape(x['observations'], (FLAGS.batch_size, -1, 96, 96, 3))
      observations = model.encode(raw_observations)
      observations_mean = observations.mean()
      observations_std_dev = observations.stddev()

      for k in range(FLAGS.batch_size):
        file_writer.write(
            structs.latent_image_rollout_to_tfrecord(
                obs_latents=observations_mean[k],
                actions=x['actions'][k],
                rewards=x['rewards'][k],
                obs_std_devs=observations_std_dev[k]))


@tf.function
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
  ds = ds.batch(FLAGS.batch_size)

  # TODO: REMOVE, THIS IS JUST FOR INITIAL TESTING!!!!!!!!!!!!!!!!!!!!!
  ds = ds.take(128)

  start = time.time()

  run(ds, FLAGS.num_shards)

  end = time.time()
  print("Took", end - start, "seconds to run.")


if __name__ == '__main__':
  app.run(main)
