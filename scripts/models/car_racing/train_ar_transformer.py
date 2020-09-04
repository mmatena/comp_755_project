"""Trains an autoregressive transformer on windows of data."""
# import functools
# import os

from absl import app

from absl import flags

# from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

# from rl755.data.car_racing import encoded_rollouts
# from rl755.data.car_racing import processing
# from rl755.models.car_racing import transformer
# from rl755.models.common import transformer as common_transformer

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)
flags.DEFINE_integer(
    "train_steps", None, "The number of steps to train for.", lower_bound=1
)

flags.DEFINE_integer(
    "batch_size", 32, "The number of sequences in each batch.", lower_bound=1
)
flags.DEFINE_integer(
    "sequence_length", 32, "Size of windows to train on.", lower_bound=1
)
# flags.DEFINE_integer(
#     "ignore_loss_prefix_size",
#     4,
#     "Ignore losses on the first few tokens.",
#     lower_bound=0,
# )
flags.DEFINE_integer(
    "num_components",
    5,
    "Number of components in the Guassian mixture model.",
    lower_bound=1,
)

flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("train_steps")


# def get_train_ds():
#     # We need the `+1` due to how we are processing the sequences.
#     ds = encoded_rollouts.random_rollout_slices(slice_size=FLAGS.sequence_length + 1)
#     ds = ds.map(
#         functools.partial(
#             transformer.to_ar_inputs_and_targets,
#             sequence_length=FLAGS.sequence_length,
#             sample=True,
#         ),
#         num_parallel_calls=tf.data.experimental.AUTOTUNE,
#     )
#     ds = processing.standard_dataset_prep(
#         ds, batch_size=FLAGS.batch_size, repeat=True, shuffle_buffer_size=1000
#     )
#     return ds


def main(_):
    # model_dir = FLAGS.model_dir

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_visible_devices(gpu, "GPU")
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Training on {len(gpus)} GPUS.")

    # # TODO(mmatena): Make these settable or inferred from the data. These correspond to BERT Base
    # output_size = 32
    # num_attention_heads = 12
    # hidden_size = 768
    # transformer_params = TransformerEncoderLayer.Params(
    #     num_layers=12,
    #     hidden_size=hidden_size,
    #     hidden_dropout=0.1,
    #     intermediate_size=4 * hidden_size,
    #     intermediate_activation="gelu",
    #     num_heads=num_attention_heads,
    #     size_per_head=int(hidden_size / num_attention_heads),
    # )

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model = common_transformer.AutoregressiveTransformer(
    #         transformer_params,
    #         output_size=output_size,
    #         num_components=FLAGS.num_components,
    #     )
    #     model.compile(
    #         loss=model.nll_loss(global_batch_size=FLAGS.batch_size), optimizer="adam"
    #     )

    # ds = get_train_ds()

    # model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=os.path.join(model_dir, "model.hdf5"),
    #     save_freq=1000,
    #     save_weights_only=True,
    #     save_best_only=False,
    # )

    # model.fit(
    #     ds,
    #     epochs=1,
    #     steps_per_epoch=FLAGS.train_steps,
    #     callbacks=[model_checkpoint_cb],
    # )


if __name__ == "__main__":
    app.run(main)
