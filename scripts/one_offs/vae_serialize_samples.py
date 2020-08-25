"""Created purely because I couldn't figure out how to get
this work on longleaf's notebook."""


from absl import app
from absl import flags
import tensorflow as tf

from rl755.models.car_racing import saved_models

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_samples', 1,
                     'The number of times to sample from the VAE.',
                     lower_bound=1)


def main(_):
  vae = saved_models.raw_rollout_vae_32ld()
  z = vae.sample_unconditionally(FLAGS.num_samples)
  z = tf.cast(255.0 * z, tf.unit8)
  serial = tf.io.serialize_tensor(z)
  print(serial.numpy())


if __name__ == '__main__':
  app.run(main)
