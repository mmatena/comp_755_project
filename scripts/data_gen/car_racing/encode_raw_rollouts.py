"""Train a VAE on random images collected from the rollouts."""
from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import raw_rollouts

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None,
                    'The path to the encoder checkpoint')

flags.DEFINE_integer('batch_size', 8,
                     'The number of rollouts in each batch.',
                     lower_bound=1)

flags.mark_flag_as_required('checkpoint_path')


@tf.function
def run(ds):
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = tf.keras.models.load_model(FLAGS.checkpoint_path)

    ds = mirrored_strategy.experimental_distribute_dataset(ds)

    for x in ds:
      # TODO(mmatena): This is tailored to VAEs. Handle non-VAE encoders.
      observations = model.encode(x['observations'])
      observations_mean = observations.mean()
      observations_std_dev = observations.stddev()


def main(_):
  ds = raw_rollouts.get_raw_rollouts_ds(process_observations=True)
  ds = ds.batch(FLAGS.batch_size)
  run(ds)


if __name__ == '__main__':
  app.run(main)
