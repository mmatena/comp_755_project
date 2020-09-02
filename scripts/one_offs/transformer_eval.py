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

SEQUENCE_LENGTH = 32


def get_metrics():
    obs_mse = transformer.observation_only_metric(
        tf.keras.metrics.MeanSquaredError(), prefix_size=4
    )
    obs_mse.__name__ = "obs_mse"
    reward_mse = transformer.reward_only_metric(
        tf.keras.metrics.MeanSquaredError(), prefix_size=4
    )
    reward_mse.__name__ = "reward_mse"
    return [obs_mse, reward_mse]


def get_ds():
    # NOTE: This won't be deterministic.
    ds = encoded_rollouts.random_rollout_slices(
        slice_size=SEQUENCE_LENGTH + 1, split="validation"
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
    model = getattr(saved_models, FLAGS.model)(k=100, corpus_size=None, lambda_knn=0.2)
    model.return_layer_outputs = False
    model.compile(optimizer="adam", loss=loss_fn, metrics=get_metrics())
    model.evaluate(ds)


if __name__ == "__main__":
    app.run(main)

# # These had the distance softmax wrong!!
# # 100k, 0.2, k=10: loss: 0.3411 - obs_mse: 0.3486 - reward_mse: 0.1190
# # 100k, 0.2, k=10: loss: 0.3407 - obs_mse: 0.3471 - reward_mse: 0.1227
# # 100k, 0.0, k=10: loss: 0.3405 - obs_mse: 0.3463 - reward_mse: 0.1274
# # None, 0.2, k=10: loss: 0.3414 - obs_mse: 0.3478 - reward_mse: 0.1234
# # None, 0.2, k=100:

# 100k, 0.2, k=10: loss: 0.3401 - obs_mse: 0.3466 - reward_mse: 0.1204
