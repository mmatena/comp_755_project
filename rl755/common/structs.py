"""Common structs."""
from collections import namedtuple

import tensorflow as tf

# A rollout step containing an observation, action, and reward.
RolloutStep = namedtuple("RolloutStep", ['o', 'a', 'r', 'step'])


class Rollout:
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


def _to_float_feature_list(array, process_item_fn):
  """Converts a 1-d float array to a tf.train.FeatureList."""
  features = []
  for a in array:
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=process_item_fn(a)))
    features.append(feature)
  return tf.train.FeatureList(feature=features)


def to_tf_record(rollout):
  """Converts a rollout to the equivalent tf record."""
  obs_features = []
  for obs in rollout.obs_l:
    obs_str = tf.io.serialize_tensor(obs).numpy()
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[obs_str]))
    obs_features.append(feature)
  obs_features = tf.train.FeatureList(feature=obs_features)

  return tf.train.SequenceExample(feature_lists=tf.train.FeatureLists(feature_list={
      "observations": obs_features,
      "actions": _to_float_feature_list(rollout.action_l, lambda a: a),
      "rewards": _to_float_feature_list(rollout.reward_l, lambda r: [r]),
  }))
