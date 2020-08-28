from absl import app
from absl import flags
import tensorflow as tf

from rl755.data.car_racing import encoded_rollouts
from rl755.models.car_racing import saved_models

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_examples", 512, "The number of examples to evaluate on.", lower_bound=1
)
flags.DEFINE_integer("batch_size", 128, "Batch size for eval.", lower_bound=1)
flags.DEFINE_integer("sequence_length", 32, "Size of windows for eval.", lower_bound=1)


def ignore_prefix_loss(loss_fn, prefix_size):
    def fn(y_true, y_pred):
        # We assume that the sequence dimension is the second dimension.
        y_true = y_true[:, prefix_size:]
        y_pred = y_pred[:, prefix_size:]
        return loss_fn(y_true, y_pred)

    return fn


def _to_inputs_and_targets(x):
    # o[i], a[i], r[i-1] => o[i+1], r[i]
    # TODO(mmatena): Make sure this is correct!
    r = tf.expand_dims(x["rewards"], axis=-1)
    a = x["actions"]
    o = x["observations"]
    inputs = tf.concat(
        [o[:-1], a[:-1], tf.concat([[[0.0]], r[:-2]], axis=0)],
        axis=-1,
    )
    targets = tf.concat([o[1:], r[:-1]], axis=-1)
    # TODO(mmatena): Don't hardcode these shapes
    inputs = tf.reshape(inputs, [FLAGS.sequence_length, 32 + 4 + 1])
    targets = tf.reshape(targets, [FLAGS.sequence_length, 32 + 1])
    return inputs, targets


def get_ds():
    # NOTE: This won't be deterministic.
    ds = encoded_rollouts.random_rollout_slices(
        slice_size=FLAGS.sequence_length + 1, split="validation"
    )
    ds = ds.take(FLAGS.num_examples)
    ds = ds.map(
        _to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = ds.batch(FLAGS.batch_size)
    return ds


def main(_):
    ds = get_ds()

    loss_fn = ignore_prefix_loss(
        tf.keras.losses.MeanSquaredError(), prefix_size=FLAGS.ignore_loss_prefix_size
    )

    model = saved_models.encoded_rollout_transformer()
    model.compile(optimizer="adam", loss=loss_fn)
    model.evaluate(ds)


if __name__ == "__main__":
    app.run(main)
