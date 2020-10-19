"""Policies that are learned via CMA-ES."""
import pickle

import numpy as np
import tensorflow as tf

from rl755.data_gen import gym_rollouts


class LearnedPolicy(object):
    @staticmethod
    def from_flat_arrays(array, in_size, out_size):
        raise NotImplementedError()

    @staticmethod
    def get_parameter_count(in_size, out_size):
        raise NotImplementedError()

    @staticmethod
    def deserialize(bytes_str):
        raise NotImplementedError()

    def sample_action(self, inputs):
        raise NotImplementedError()

    def in_size(self):
        raise NotImplementedError()

    def out_size(self):
        raise NotImplementedError()

    def serialize(self):
        raise NotImplementedError()


class LinearPolicy(LearnedPolicy):
    @staticmethod
    def from_flat_arrays(array, in_size, out_size):
        array = np.array(array)
        w, b = array[:, :-out_size], array[:, -out_size:]
        w = np.reshape(w, [-1, out_size, in_size])
        return LinearPolicy(w=w, b=b)

    @staticmethod
    def get_parameter_count(in_size, out_size):
        return in_size * out_size + out_size

    @staticmethod
    def deserialize(bytes_str):
        w, b = pickle.loads(bytes_str)
        return LinearPolicy(w=w, b=b)

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self._out_size = b.shape[-1]
        self._in_size = w.shape[-1]

    def sample_action(self, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        action = np.einsum("...jk,...k->...j", self.w, inputs) + self.b
        action = np.reshape(action, [-1, self._out_size])
        return action

    def in_size(self):
        return self._in_size

    def out_size(self):
        return self._out_size

    def serialize(self):
        return pickle.dumps((self.w, self.b))


class PolicyWrapper(gym_rollouts.Policy):
    # Note: This expects everything to have a batch dimension.
    def __init__(self, vision_model, sequence_model, learned_policy, max_seqlen):
        self.vision_model = vision_model
        self.sequence_model = sequence_model
        self.learned_policy = learned_policy
        self.max_seqlen = max_seqlen

    def initialize(self, env, max_steps, **kwargs):
        self.encoded_obs = []

    def _ensure_sequence_length(self, x):
        x = x[..., -self.max_seqlen :, :]
        diff = self.max_seqlen - x.shape[-2]
        batch_dims = x.shape[:-2]
        if diff:
            mask = tf.concat(
                [tf.ones(x.shape[:-1]), tf.zeros(batch_dims + [diff])], axis=-1
            )
            padding = tf.zeros(batch_dims + [diff, x.shape[-1]], dtype=tf.float32)
            x = tf.concat([x, padding], axis=-2)
            return x, mask
        else:
            return x, None

    def _create_sequence_model_inputs(self, rollout):
        observations = self.encoded_obs[-self.max_seqlen :]
        actions = rollout.action_l[-self.max_seqlen :]
        nonpadding_seqlen = len(observations)

        observations = tf.stack(observations, axis=-2)
        actions = tf.stack(actions, axis=-2)
        actions = tf.cast(actions, tf.float32)

        inputs = tf.concat([observations, actions], axis=-1)
        inputs, mask = self._ensure_sequence_length(inputs)
        return inputs, mask, nonpadding_seqlen

    def sample_action(self, obs, step, rollout, **kwargs):
        obs = tf.cast(obs, tf.float32) / 255.0
        enc_obs = self.vision_model.encode_tensor(obs)
        # TODO(mmatena): Handle this case better.
        if step == 0:
            self.encoded_obs.append(enc_obs)
            return self.learned_policy.sample_action(
                np.zeros([enc_obs.shape[0], self.learned_policy.in_size()])
            )
        inputs, mask, nonpadding_seqlen = self._create_sequence_model_inputs(rollout)
        hidden_state = self.sequence_model.get_hidden_representation(
            inputs, mask=mask, position=nonpadding_seqlen - 1
        )
        self.encoded_obs.append(enc_obs)
        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)
        action = self.learned_policy.sample_action(policy_input)
        return action
