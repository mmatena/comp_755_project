"""Evaluate an autoregressive transformer on a validation set."""
import functools

from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import encoded_rollouts
from rl755.models.car_racing import saved_models
from rl755.models.car_racing import transformer

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_examples", 512, "The number of examples to evaluate on.", lower_bound=1
)
flags.DEFINE_integer("batch_size", 128, "Batch size for eval.", lower_bound=1)

SEQUENCE_LENGTH = 32


def get_metrics():
    obs_mse = transformer.observation_only_metric(
        tf.keras.metrics.MeanSquaredError(name="obs_mse")
    )
    rewward_mse = transformer.observation_only_metric(
        tf.keras.metrics.MeanSquaredError(name="rewward_mse")
    )
    return [obs_mse, rewward_mse]


def get_ds():
    # NOTE: This won't be deterministic.
    ds = encoded_rollouts.random_rollout_slices(
        slice_size=SEQUENCE_LENGTH + 1, split="validation"
    )
    ds = ds.take(FLAGS.num_examples)
    ds = ds.map(
        functools.partial(
            transformer.to_ar_inputs_and_targets, sequence_length=SEQUENCE_LENGTH
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.batch(FLAGS.batch_size)
    return ds


def main(_):
    ds = get_ds()

    loss_fn = transformer.ignore_prefix_loss(
        tf.keras.losses.MeanSquaredError(), prefix_size=4
    )

    model = saved_models.encoded_rollout_transformer()
    # model.compile(optimizer="adam", loss=loss_fn, metrics=get_metrics())
    model.compile(optimizer="adam", loss=loss_fn)
    model.evaluate(ds)


if __name__ == "__main__":
    app.run(main)
