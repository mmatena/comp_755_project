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
        return loss_fn(y_true, y_pred)

    return fn


def to_ar_inputs_and_targets(
    x, sequence_length, latent_size=32, action_size=3, sample=False
):
    """Given a slice of a rollout, convert it to inputs and targets for autoregressive modelling.

    We do it like this:
        o[i], a[i] => o[i+1] - o[i]
    The goal is to predict the change in observation given the action taken and the previous states
    and actions.
    """
    # TODO(mmatena): Make sure this is correct!
    a = x["actions"][:, :action_size]
    o = x["observations"]
    # TODO(mmatena): This messes up training for some reason. I think the std devs are bad.
    # if sample:
    #     o += x["observation_std_devs"] * tf.random.normal(shape=tf.shape(o))
    inputs = tf.concat(
        [o[:-1], a[:-1]],
        axis=-1,
    )
    targets = o[1:] - o[:-1]
    inputs = tf.reshape(inputs, [sequence_length, latent_size + action_size])
    targets = tf.reshape(targets, [sequence_length, latent_size])
    return inputs, targets


def observation_only_metric(metric_fn, latent_size=32, prefix_size=0):
    """Computes a metric only on the observations."""

    def fn(y_true, y_pred):
        y_true = y_true[:, prefix_size:, :latent_size]
        y_pred = y_pred[:, prefix_size:, :latent_size]
        return metric_fn(y_true, y_pred)

    return fn


def reward_only_metric(metric_fn, prefix_size=0):
    """Computes a metric only on the rewards."""

    def fn(y_true, y_pred):
        y_true = y_true[:, prefix_size:, -1:]
        y_pred = y_pred[:, prefix_size:, -1:]
        return metric_fn(y_true, y_pred)

    return fn
