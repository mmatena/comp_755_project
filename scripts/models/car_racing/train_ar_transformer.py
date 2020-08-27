"""Trains an autoregressive transformer on windows of data."""
from absl import app
from absl import flags
import tensorflow as tf

flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)
flags.DEFINE_integer(
    "train_steps", None, "The number of steps to train for.", lower_bound=1
)

flags.DEFINE_integer(
    "batch_size", 256, "The number of images in each batch.", lower_bound=1
)
flags.DEFINE_integer(
    "sequence_length", 32, "Size of windows to train on.", lower_bound=1
)
flags.DEFINE_integer(
    "ignore_loss_prefix_size",
    4,
    "Ignore losses on the first few tokens.",
    lower_bound=0,
)

flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("train_steps")


def ignore_prefix_loss(loss_fn, prefix_size):
    def fn(y_true, y_pred):
        # We assume that the sequence dimension is the second dimension.
        y_true = y_true[:, prefix_size:]
        y_pred = y_pred[:, prefix_size:]
        return loss_fn(y_true, y_pred)

    return fn


def main(_):
    pass
