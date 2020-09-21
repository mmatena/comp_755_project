"""Trains an autoregressive model on encoded rollouts."""
import functools
import os

from absl import app
from absl import flags

import tensorflow as tf

from rl755 import data as data_module
from rl755 import models as models_module
from rl755.data.common import processing
from rl755.environments import Environments

_ENV_NAME_TO_ENV = Environments.__members__

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)
flags.DEFINE_integer(
    "train_steps", None, "The number of steps to train for.", lower_bound=1
)

flags.DEFINE_enum(
    "environment",
    None,
    _ENV_NAME_TO_ENV.keys(),
    "Which environment we are using rollouts from.",
)
flags.DEFINE_string("model", None, "")
flags.DEFINE_string("rollouts_dataset", None, "")

flags.DEFINE_integer(
    "batch_size", 32, "The number of sequences in each batch.", lower_bound=1
)
flags.DEFINE_integer(
    "sequence_length", 32, "Size of windows to train on.", lower_bound=1
)
flags.DEFINE_float("learning_rate", 1e-3, "")
flags.DEFINE_integer(
    "save_every_n_steps",
    1000,
    "The model will save a checkpoint every this many steps. Should divide train_steps.",
)

flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("train_steps")
flags.mark_flag_as_required("rollouts_dataset")
flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model")


def get_environment():
    return _ENV_NAME_TO_ENV[FLAGS.environment]


def get_model(environment):
    model = getattr(models_module, environment.folder_name)
    for s in FLAGS.model.split("."):
        model = getattr(model, s)
    return model()


def get_train_dataset(environment):
    m = getattr(data_module, environment.folder_name).encoded_rollouts
    ds = getattr(m, FLAGS.rollouts_dataset)().get_autoregressive_slices(
        sequence_length=FLAGS.sequence_length,
        sample_observations=False,
        difference_targets=False,
        split="train",
    )
    return processing.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)


def main(_):
    model_dir = FLAGS.model_dir
    environment = get_environment()

    file_writer = tf.summary.create_file_writer(model_dir)
    file_writer.set_as_default()

    train_ds = get_train_dataset(environment)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = get_model(environment)

        # Hardcoded parameters taken from the "Attention is all you need" paper.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "model-{epoch:03d}.hdf5"),
            save_best_only=False,
            save_weights_only=True,
        )
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
        callbacks = [model_checkpoint_cb, tensorboard_cb]

        model.compile(optimizer=optimizer, loss=model.get_loss_fn())

    steps_per_epoch = FLAGS.save_every_n_steps
    model.fit(
        train_ds,
        epochs=FLAGS.train_steps // steps_per_epoch,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    app.run(main)
