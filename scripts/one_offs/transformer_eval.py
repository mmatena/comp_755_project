"""Evaluate an autoregressive transformer on a validation set."""
import ast
import functools

from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import encoded_rollouts
from rl755.models.car_racing import saved_models
from rl755.models.car_racing import transformer

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_examples", 32 * 1024, "The number of examples to evaluate on.", lower_bound=1
)
flags.DEFINE_integer("batch_size", 256, "Batch size for eval.", lower_bound=1)

flags.DEFINE_string("model", "encoded_rollout_transformer", "Name of model to use.")
flags.DEFINE_string(
    "model_kwargs", "{}", "Kwargs dict literal for intstantiating the model."
)

flags.DEFINE_string("split", "validation", "The split to evaluate on.")

SEQUENCE_LENGTH = 32


def get_ds():
    # NOTE: This won't be deterministic.
    ds = encoded_rollouts.random_rollout_slices(
        slice_size=SEQUENCE_LENGTH + 1, split=FLAGS.split
    )
    ds = ds.repeat()
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
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    ds = get_ds()

    loss_fn = transformer.ignore_prefix_loss(
        tf.keras.losses.MeanSquaredError(), prefix_size=4
    )

    # model = getattr(saved_models, FLAGS.model)(**ast.literal_eval(FLAGS.model_kwargs))
    # model = getattr(saved_models, FLAGS.model)(k=10, corpus_size=100000, lambda_knn=1.0)
    # model = getattr(saved_models, FLAGS.model)(k=10, corpus_size=100000, lambda_knn=0.2)
    model = getattr(saved_models, FLAGS.model)()
    model.return_layer_outputs = False
    model.compile(optimizer="adam", loss=loss_fn)
    model.evaluate(ds)


if __name__ == "__main__":
    app.run(main)
