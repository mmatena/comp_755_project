"""Created purely because I couldn't figure out how to get
this work on longleaf's notebook."""
import itertools

from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import raw_rollouts
from rl755.models.car_racing import saved_models

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_samples', 1,
                     'The number of times to sample from the VAE.',
                     lower_bound=1)

flags.DEFINE_bool('unconditional', True,
                  'Whether to condition on real images or not.')


def float_to_uint(x):
  return tf.cast(255.0 * x, tf.uint8)


def main(_):
  vae = saved_models.raw_rollout_vae_32ld()
  if FLAGS.unconditional:
    images = vae.sample_unconditionally(FLAGS.num_samples)
  else:
    inputs = [
        x for x in itertools.islice(raw_rollouts.random_rollout_observations(), FLAGS.num_samples)]
    inputs = tf.stack(inputs, axis=0)
    # TODO(mmatena): Compare .sample() vs .mean().
    z = vae.encode(inputs).mean()
    images = vae.decode(z)

    print(tf.io.serialize_tensor(inputs).numpy())
    for _ in range(10):
      print("_")

  images = float_to_uint(images)
  print(tf.io.serialize_tensor(images).numpy())


if __name__ == '__main__':
  app.run(main)
