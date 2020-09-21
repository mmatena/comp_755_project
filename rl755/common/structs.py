"""Common structs."""
from collections import namedtuple

import tensorflow as tf

# A rollout step containing an observation, action, and reward.
RolloutStep = namedtuple("RolloutStep", ["o", "a", "r", "step"])


class Rollout:
    """A generic class representing a rollout of some environment."""

    def __init__(self):
        # The observation at step i.
        self.obs_l = []
        # The action taken at step i.
        self.action_l = []
        # The reward received in the transition s_i -> s_{i+1}.
        self.reward_l = []
        # Whether the episode reached a terminal state. Ignore
        # for non-episodic tasks.
        self.done = False


def _to_float_feature_list(array, process_item_fn=lambda x: x):
    """Converts an array of floats to a tf.train.FeatureList.

    The `array` must be representable as a rank 2 tensor. Note that this trivially
    includes rank 1 and rank 0 tensors.

    Args:
        array: an iterable object
        process_item_fn: a function to be applied to each item in the array. Must return
            a 1-dimensional list-like object of floats.
    Returns:
        A tf.train.FeatureList corresponding to `array`.
    """
    features = []
    for a in array:
        feature = tf.train.Feature(
            float_list=tf.train.FloatList(value=process_item_fn(a))
        )
        features.append(feature)
    return tf.train.FeatureList(feature=features)


def raw_rollout_to_tfrecord(rollout):
    """Converts a rollout to the equivalent tfrecord.

    Args:
        rollout: a rl755.common.structs.Rollout, the rollout to convert
    Returns:
        A tf.train.SequenceExample corresponding to `rollout`.
    """
    obs_features = []
    for obs in rollout.obs_l:
        obs_str = tf.io.serialize_tensor(obs).numpy()
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[obs_str]))
        obs_features.append(feature)
    obs_features = tf.train.FeatureList(feature=obs_features)

    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                "observations": obs_features,
                "actions": _to_float_feature_list(rollout.action_l, lambda a: a),
                "rewards": _to_float_feature_list(rollout.reward_l, lambda r: [r]),
            }
        )
    )


def key_value_to_tfrecord(key, value):
    """Converts a key value pair to the equivalent tfrecord.

    Args:
        key: a 1-d float tensor
        value: a 1-d float tensor
    Returns:
        A tf.train.Example.
    """
    features = {
        "key": tf.train.Feature(float_list=tf.train.FloatList(value=key)),
        "value": tf.train.Feature(float_list=tf.train.FloatList(value=value)),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def tfrecord_to_key_value(record, key_size=768, value_size=33):
    # TODO(mmatena): Add docs
    features = {
        "key": tf.io.FixedLenFeature([key_size], tf.float32),
        "value": tf.io.FixedLenFeature([value_size], tf.float32),
    }
    return tf.io.parse_single_example(record, features)


def encoded_rollout_to_tfrecord(example):
    """Converts a rollout with encoded features into a tfrecord.

    Args:
        example: dict[str, tf.Tensor], the rollout. We assume all features are rank 2, tf.float32
            tensors except for "rewards", which is a rank 1, tf.float32 tensor. Note that these
            ranks include the time dimension.
    Returns:
        The example as a tf.train.SequenceExample.
    """
    feature_list = {
        k: _to_float_feature_list(v) for k, v in example.items() if k != "rewards"
    }
    feature_list["rewards"] = _to_float_feature_list(example["rewards"], lambda r: [r])
    return tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )
