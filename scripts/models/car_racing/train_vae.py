"""Train a VAE on random images collected from the rollouts."""
import os

from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import raw_rollouts
from rl755.models.car_racing.vae import Vae

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', None,
                    'The directory to write checkpoints and logs to.')
flags.DEFINE_integer('train_steps', None,
                     'The number of steps to train for.',
                     lower_bound=1)

flags.DEFINE_float('beta', 1.0,
                   'The weight of the KL loss term in the VAE.')
flags.DEFINE_integer('batch_size', 256,
                     'The number of images in each batch.',
                     lower_bound=1)
flags.DEFINE_integer('latent_size', 32,
                     'The dimensionality of the latent variable.',
                     lower_bound=1)


flags.mark_flag_as_required('model_dir')
flags.mark_flag_as_required('train_steps')


def main(_):
  model_dir = FLAGS.model_dir

  ds = raw_rollouts.random_rollout_observations(obs_per_rollout=8, shuffle_files=True)
  ds = raw_rollouts.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)

  model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(model_dir, "model.hdf5"),
      save_freq=1000,
      save_best_only=False)
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir)

  vae = Vae(latent_dim=FLAGS.latent_size, beta=FLAGS.beta)
  vae.compile(optimizer="adam")

  vae.fit(ds, epochs=1, steps_per_epoch=FLAGS.train_steps,  verbose=2,
          callbacks=[model_checkpoint_cb, tensorboard_cb])


if __name__ == '__main__':
  app.run(main)