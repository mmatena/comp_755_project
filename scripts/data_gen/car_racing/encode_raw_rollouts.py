"""Train a VAE on random images collected from the rollouts."""
import os
import time

from absl import app
from absl import flags
import ray
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
  posterior = model.encode(x)
  return posterior.mean(), posterior.stddev()


@ray.remote
def run_shard(model, ds, shard_index, num_shards, out_dir, out_name):
  model = load_model(model)
  ds = ds.shard(num_shards=num_shards, index=shard_index)

  filepath = os.path.join(out_dir,
                          misc.sharded_filename(out_name,
                                                shard_index=shard_index,
                                                num_shards=num_shards))
  with tf.io.TFRecordWriter(filepath) as file_writer:
    for x in ds:
      # TODO(mmatena): This is tailored to VAEs. Handle non-VAE encoders.
      raw_observations = tf.reshape(x['observations'], (-1, 96, 96, 3))
      mean, std_dev = encode(model, raw_observations)
      file_writer.write(
          structs.latent_image_rollout_to_tfrecord(
              obs_latents=mean,
              actions=x['actions'],
              rewards=x['rewards'],
              obs_std_devs=std_dev).SerializeToString())


def run(ds, num_shards):
  futures = [
      run_shard(FLAGS.model, ds=ds,
                shard_index=i, num_shards=num_shards,
                out_dir=FLAGS.out_dir, out_name=FLAGS.out_name)
      for i in range(num_shards)
  ]
  ray.get(futures)


def main(_):
  ray.init()
  ds = raw_rollouts.get_raw_rollouts_ds(process_observations=True)

  # TODO: REMOVE, THIS IS JUST FOR INITIAL TESTING!!!!!!!!!!!!!!!!!!!!!
  ds = ds.take(8)

  start = time.time()

  run(ds, FLAGS.num_shards)

  end = time.time()
  print("Took", end - start, "seconds to run.")


if __name__ == '__main__':
  app.run(main)
