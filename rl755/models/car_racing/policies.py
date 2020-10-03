"""Policies for running in the OpenAI Gym.


Here is the information about a 3-d action vector:
    a[0]: [-1, 1], steering
    a[1]: [0, 1], gas
    a[2]: [0, 1], brakes
Values below/above the range are clipped to the min/max of the range, respectively.

"""
from noise import pnoise1
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rl755.data_gen import gym_rollouts

tfd = tfp.distributions


class HastingsRandomPolicy(gym_rollouts.Policy):
    """The random policy that Hastings Greer was using.

    Currently only supports the CarRacing-v0 environment.

    NOTE: The actions returned here have length 4 when the car racing
    only needs actions of length 3.
    """

    def __init__(self, time_scale=200, magnitude_scale=1.7):
        self._time_scale = time_scale
        self._magnitude_scale = magnitude_scale

    def initialize(self, env, max_steps, **kwargs):
        del env, kwargs
        start = np.random.random(4) * 10000
        out = []
        for i in range(2000):
            action = [
                self._magnitude_scale * pnoise1(i / self._time_scale + start_i, 5)
                for start_i in start
            ]
            out.append(action)
        self._actions = np.array(out)

    def sample_action(self, obs, step, **kwargs):
        del obs, kwargs
        return self._actions[step]


# class CarRacingPolicy(gym_rollouts.Policy):
#     def __init__(self, encoder, sequence_model, policy, max_seqlen=32):
#         self.encoder = encoder
#         self.sequence_model = sequence_model
#         self.policy = policy
#         self.max_seqlen = max_seqlen

#     def initialize(self, env, max_steps, **kwargs):
#         self.encoded_obs = []

#     def _ensure_sequence_length(self, x):
#         x = x[-self.max_seqlen :]
#         diff = self.max_seqlen - x.shape[0]
#         if diff:
#             mask = tf.concat([tf.ones(x.shape[:1]), tf.zeros([diff])], axis=0)
#             padding = tf.zeros([diff, x.shape[1]], dtype=tf.float32)
#             x = tf.concat([x, padding], axis=0)
#             return x, mask
#         else:
#             return x, None

#     def _create_inputs(self, rollout):
#         # o[i], a[i],] => o[i+1] or o[i+1] - o[i]
#         observations = self.encoded_obs[-self.max_seqlen :]
#         actions = rollout.action_l[-self.max_seqlen :]
#         inputs = tf.concat([observations, actions], axis=-1)
#         inputs, mask = self._ensure_sequence_length(inputs)
#         inputs = tf.expand_dims(inputs, axis=0)
#         mask = tf.expand_dims(mask, axis=0) if mask is not None else None
#         return inputs, mask

#     def sample_action(self, obs, step, rollout, **kwargs):
#         print(f"Step: {step}")
#         obs = tf.cast(obs, tf.float32) / 255.0
#         enc_obs = self.encoder.encode_tensor(tf.expand_dims(obs, axis=0))
#         # TODO(mmatena): Handle this case better.
#         if step == 0:
#             self.encoded_obs.append(enc_obs[0])
#             return self.policy.sample_action(np.zeros([256 + 32]))
#         inputs, mask = self._create_inputs(rollout)
#         # TODO(mmatena): This could be potentially be made hugely more efficient by reusing computations.
#         hidden_state = self.sequence_model.get_last_representation_tensor(
#             inputs, mask=mask
#         )

#         self.encoded_obs.append(enc_obs[0])

#         policy_input = tf.concat([enc_obs[0], hidden_state[0]], axis=-1)

#         action = self.policy.sample_action(policy_input)

#         return action


class CarRacingPolicy(gym_rollouts.Policy):
    # Note: This expects everything to have a batch dimension.
    def __init__(self, encoder, sequence_model, policy, max_seqlen=32):
        self.encoder = encoder
        self.sequence_model = sequence_model
        self.policy = policy
        self.max_seqlen = max_seqlen

    def initialize(self, env, max_steps, **kwargs):
        self.encoded_obs = []

    def _ensure_sequence_length(self, x):
        x = x[-self.max_seqlen :]
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

    def _create_inputs(self, rollout):
        # o[i], a[i],] => o[i+1] or o[i+1] - o[i]
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
        enc_obs = self.encoder.encode_tensor(obs)
        # TODO(mmatena): Handle this case better.
        if step == 0:
            self.encoded_obs.append(enc_obs)
            return self.policy.sample_action(np.zeros([enc_obs.shape[0], 256 + 32]))
        inputs, mask, nonpadding_seqlen = self._create_inputs(rollout)
        # TODO(mmatena): This could be potentially be made hugely more efficient by reusing computations.
        hidden_state = self.sequence_model.get_hidden_representation(
            inputs, mask=mask, position=nonpadding_seqlen - 1
        )

        self.encoded_obs.append(enc_obs)

        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)

        action = self.policy.sample_action(policy_input)

        return action
