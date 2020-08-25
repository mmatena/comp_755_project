"""Train a VAE on random images collected from the rollouts."""
import functools
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

flags.DEFINE_integer('num_outer_shards', None,
                     'The number of times this binary is being called independently.',
                     lower_bound=1)
flags.DEFINE_integer('outer_shard_index', None,
                     'The index in the runs of this binary.',
                     lower_bound=0)
flags.DEFINE_integer('num_sub_shards', None,
                     'The number of shards to generate for this part of the dataset. '
                     'A good goal is to strive for shards of size 100-200 MB.',
                     lower_bound=1)
flags.DEFINE_integer('gpu_index', None,
                     'The index of the GPU to use on multi-gpu systems.',
                     lower_bound=0)

flags.DEFINE_string('out_dir', None, 'The directory to write the tfrecords to.')
flags.DEFINE_string('out_name', None, 'Prefix to give the generated tfrecord files.')

flags.DEFINE_string('model', "raw_rollout_vae_32ld",
                    'Name of model in `saved_models` to use.')

flags.mark_flag_as_required('num_outer_shards')
flags.mark_flag_as_required('outer_shard_index')
flags.mark_flag_as_required('num_sub_shards')
flags.mark_flag_as_required('gpu_index')
flags.mark_flag_as_required('out_dir')
flags.mark_flag_as_required('out_name')


def get_dataset_files():
  files = tf.io.matching_files(raw_rollouts.TFRECORDS_PATTERN).numpy().tolist()



  # TODO: REMOVE, THIS IS JUST FOR INITIAL TESTING!!!!!!!!!!!!!!!!!!!!!
  files = files[:32]


  # Ensure a consistent order so that each file is processed exactly once.
  files.sort()
  return files


def get_dataset(outer_shard_index, num_outer_shards, sub_shard_index, num_sub_shards):
  files = get_dataset_files()
  files = misc.evenly_partition(files, num_outer_shards)[outer_shard_index]
  files = misc.evenly_partition(files, num_sub_shards)[sub_shard_index]

  files = tf.data.Dataset.from_tensor_slices(files)
  ds = files.interleave(tf.data.TFRecordDataset,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE,
                        deterministic=False)
  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  return ds.map(functools.partial(raw_rollouts.parse_fn, process_observations=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


def load_model(model_name):
  return getattr(saved_models, model_name)()


@tf.function
def encode(model, x):
  posterior = model.encode(x)
  return posterior.mean(), posterior.stddev()


def run_shard(model, out_dir, out_name,
              outer_shard_index, num_outer_shards, sub_shard_index, num_sub_shards):
  ds = get_dataset(outer_shard_index=outer_shard_index,
                   num_outer_shards=num_outer_shards,
                   sub_shard_index=sub_shard_index,
                   num_sub_shards=num_sub_shards)

  num_total_shards = num_outer_shards * num_sub_shards
  total_shard_index = num_sub_shards * outer_shard_index + sub_shard_index

  filepath = os.path.join(out_dir,
                          misc.sharded_filename(out_name,
                                                shard_index=total_shard_index,
                                                num_shards=num_total_shards))
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


def main(_):
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu_index], 'GPU')

  model = load_model(FLAGS.model)

  start = time.time()

  for i in range(FLAGS.num_sub_shards):
    run_shard(model=model,
              out_dir=FLAGS.out_dir,
              out_name=FLAGS.out_name,
              outer_shard_index=FLAGS.outer_shard_index,
              num_outer_shards=FLAGS.num_outer_shards,
              sub_shard_index=i,
              num_sub_shards=FLAGS.num_sub_shards)

  end = time.time()
  print("Took", end - start, "seconds to run.")


if __name__ == '__main__':
  app.run(main)
