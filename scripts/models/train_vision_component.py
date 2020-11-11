"""Trains a model to encode individual observations."""
import os
from pydoc import locate

from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.common import processing

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)
flags.DEFINE_integer(
    "train_steps", None, "The number of steps to train for.", lower_bound=1
)
flags.DEFINE_string(
    "environment",
    None,
    "Which environment we are encoding observations for.",
)
flags.DEFINE_string(
    "model",
    None,
    "The name of the model to use. Should be accessable via rl755.models.instances.$model. "
    "It should subclass rl755.models.vision.interface.VisionComponent or be a function returning such an object. "
    "It will be called with no arguments.",
)

flags.DEFINE_integer(
    "batch_size", 256, "The number of images in each batch.", lower_bound=1
)
flags.DEFINE_integer(
    "save_every_n_steps",
    1000,
    "The model will save a checkpoint every this many steps. Should divide train_steps.",
)
flags.DEFINE_integer(
    "obs_sampled_per_rollout",
    100,
    "The model will take this many observations at random each time it encounters a rollout.",
    lower_bound=1,
)
flags.DEFINE_float("learning_rate", 1e-3, "")
flags.DEFINE_float("color_jitter_strength", 1.0, "The strength of color jittering.")

flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("train_steps")
flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model")


def get_model():
    model = locate(f"rl755.models.vision.instances.{FLAGS.model}")
    return model()


def get_final_ds_representation(x):
    return x["observation"], x["observation"]


def get_aug_ds_representation(x):
    image = x["observation"]
    xs = []
    for _ in range(2):  # Two transformations
        xs.append(processing.augment_for_train(image, 64, 64))
    image = tf.concat(xs, axis=-1)
    return image, 0.0


def get_train_dataset():
    dsb_cls = locate(f"rl755.data.envs.{FLAGS.environment}.RawRollouts")
    ds = dsb_cls().random_rollout_observations(
        obs_sampled_per_rollout=FLAGS.obs_sampled_per_rollout
    )

    ds = processing.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)
    return ds.map(
        get_final_ds_representation,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # # for contrastive learning
    # if "clr" in FLAGS.model:
    #     ds = ds.map(
    #         get_aug_ds_representation, num_parallel_calls=tf.data.experimental.AUTOTUNE
    #     )
    #     ds = processing.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)
    #     return ds
    # # for vae training
    # elif "vae" in FLAGS.model:
    #     ds = processing.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)
    #     return ds.map(
    #         get_final_ds_representation,
    #         num_parallel_calls=tf.data.experimental.AUTOTUNE,
    #     )


def main(_):
    model_dir = FLAGS.model_dir

    file_writer = tf.summary.create_file_writer(model_dir)
    file_writer.set_as_default()

    train_ds = get_train_dataset()

    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "model-{epoch:03d}.hdf5"),
        save_best_only=False,
        save_weights_only=True,
    )
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
    callbacks = [model_checkpoint_cb, tensorboard_cb]

    model = get_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

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
