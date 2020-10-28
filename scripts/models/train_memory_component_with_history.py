"""Trains an autoregressive model on encoded rollouts with access to the full history
   of an episode.
"""
import functools
import os
from pydoc import locate

from absl import app
from absl import flags

import tensorflow as tf

from rl755.data.common import processing
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder

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
    "Which environment we are using rollouts from.",
)
flags.DEFINE_string("model", None, "")
flags.DEFINE_string("rollouts_dataset", None, "")

flags.DEFINE_integer(
    "batch_size", 256, "The number of sequences in each batch.", lower_bound=1
)
flags.DEFINE_integer(
    "sequence_length", 32, "Size of windows to train on.", lower_bound=1
)
flags.DEFINE_integer("max_history_length", 512, "", lower_bound=1)

flags.DEFINE_float("learning_rate", 1e-4, "")


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


# def get_model():
#     from rl755.models.memory.retrieval import EpisodicRetriever
#     from rl755.models.memory.trained import caveflyer

#     print("TODO: configure all this with flags and stuff")
#     prediction_network = caveflyer.deterministic_transformer_32dm_32di(
#         name="prediction"
#     )
#     query_network = caveflyer.deterministic_transformer_32dm_32di(name="query")
#     key_network = caveflyer.deterministic_transformer_32dm_32di(name="key")

#     sequence_length = 32
#     key_size = 32
#     history_stride = sequence_length // 2
#     num_retrieved = 4

#     model = EpisodicRetriever(
#         prediction_network=prediction_network,
#         key_network=key_network,
#         query_network=query_network,
#         key_size=key_size,
#         history_stride=history_stride,
#         num_retrieved=num_retrieved,
#     )
#     return model
def get_model():
    from rl755.models.memory.trained import caveflyer

    prediction_network = caveflyer.deterministic_transformer_32dm_32di()

    class Model(tf.keras.Model):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def call(self, inputs, mask=None, training=None):
            return self.net(inputs["inputs"], mask=mask, training=training)[..., -1, :]

        def get_loss_fn(self):
            """Train using a MSE loss."""
            return tf.keras.losses.MeanSquaredError()

    return Model(prediction_network)


def get_train_dataset():
    dsb_cls = locate(f"rl755.data.envs.{FLAGS.environment}.{FLAGS.rollouts_dataset}")
    dsb = dsb_cls()

    ds = dsb.get_autoregressive_slices_with_full_history(
        sequence_length=FLAGS.sequence_length,
        max_history_length=FLAGS.max_history_length,
        split="train",
    )

    return processing.standard_dataset_prep(ds, batch_size=FLAGS.batch_size)


def main(_):
    model_dir = FLAGS.model_dir

    file_writer = tf.summary.create_file_writer(model_dir)
    file_writer.set_as_default()

    train_ds = get_train_dataset()

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():

        model = get_model()

        # Hardcoded parameters taken from the "Attention is all you need" paper.
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        )

        print("TODO: Add model checkpoint callback.")
        # model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(model_dir, "model-{epoch:03d}.hdf5"),
        #     save_best_only=False,
        #     save_weights_only=True,
        # )
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=model_dir)
        callbacks = [
            # model_checkpoint_cb,
            tensorboard_cb
        ]

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
