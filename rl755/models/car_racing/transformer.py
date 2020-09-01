"""Car racing specific transformer code."""
import tensorflow as tf

from rl755.models.common import transformer as common_transformer


# TODO(mmatena): Put some special stuff here.
class AutoregressiveFwdOr(common_transformer.AutoregressiveTransformer):
    pass

    # def predict_next(self, rollout, action):
    #     # TODO(mmatena): Return maybe change the inputs here.
    #     # TODO(mmatena): Return next state, next reward
    #     pass


def ignore_prefix_loss(loss_fn, prefix_size):
    """Ignore the losses on the first few tokens.

    The idea is that the model might not have enough information to accurately
    predict the next frame when modeling autoregressively. One reason is that you
    cannot tell the velocity from a single frame.

    Args:
        loss_fn: a loss function compatible with keras
        prefix_size: a non-negative integer, the loss on the first `prefix_size` tokens
            will be ignored
    Returns:
        A loss function compatible with keras.
    """

    def fn(y_true, y_pred):
        # We assume that the sequence dimension is the second dimension.
        y_true = y_true[:, prefix_size:]
        y_pred = y_pred[:, prefix_size:]
        return loss_fn(y_true, y_pred, sample_weight=None)

    return fn


def to_ar_inputs_and_targets(x, sequence_length, latent_size=32, action_size=4):
    """Given a slice of a rollout, convert it to inputs and targets for autoregressive modelling.

    We do it like this:
        o[i], a[i], r[i-1] => o[i+1], r[i]
    The goal is to predict the next observation and reward given the action taken and the previous states
    and rewards.
    """
    # TODO(mmatena): Make sure this is correct!
    r = tf.expand_dims(x["rewards"], axis=-1)
    a = x["actions"]
    o = x["observations"]
    inputs = tf.concat(
        [o[:-1], a[:-1], tf.concat([[[0.0]], r[:-2]], axis=0)],
        axis=-1,
    )
    targets = tf.concat([o[1:], r[:-1]], axis=-1)
    inputs = tf.reshape(inputs, [sequence_length, latent_size + action_size + 1])
    targets = tf.reshape(targets, [sequence_length, latent_size + 1])
    return inputs, targets


def observation_only_metric(metric_fn, latent_size=32):
    """Computes a metric only on the observations."""

    def fn(y_true, y_pred):
        y_true = y_true[..., :latent_size]
        # y_true = tf.reshape(y_true, [-1, latent_size])

        y_pred = y_pred[..., :latent_size]
        # y_pred = tf.reshape(y_pred, [-1, latent_size])

        return metric_fn(y_true, y_pred)

    return fn


def reward_only_metric(metric_fn, latent_size=32):
    """Computes a metric only on the rewards."""

    def fn(y_true, y_pred):
        y_true = y_true[..., -1:]
        # y_true = tf.reshape(y_true, [-1, 1])

        y_pred = y_pred[..., -1:]
        # y_pred = tf.reshape(y_pred, [-1, 1])

        return metric_fn(y_true, y_pred)

    return fn
