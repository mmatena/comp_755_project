"""Policies.

Note that these don't contain any learnable parameters themselves.
Instead they let us interface components with the gym3 environment.
"""
import gym3
import numpy as np
import tensorflow as tf


class Policy(object):
    """Abstract base class for policies"""

    def initialize(self, env, max_steps):
        """Initializes the policy upon starting a new episode.

        Args:
            env: the OpenAI Gym3 environment we will be running on
            max_steps: a postive integer, the maximum number of steps the
                rollout will go on for
        """
        raise NotImplementedError()

    def sample_action(self, rollout_state):
        """Generate an action.

        Args:
            rollout: the RolloutState object corresponding to this run
        Returns:
            A np array with dtype int32 and shape [num_env] with values in the range [0, 15).
        """
        raise NotImplementedError()


class UniformRandomPolicy(Policy):
    """A policy that samples from the action space uniformly and IID."""

    def initialize(self, env, max_steps):
        del max_steps
        self.env = env

    def sample_action(self, rollout_state):
        del rollout_state
        return gym3.types_np.sample(self.env.ac_space, bshape=(self.env.num,))


class PolicyWrapper(Policy):
    # Note: TODO: Add docs
    def __init__(self, vision_model, memory_model, learned_policy, max_seqlen):
        self.vision_model = vision_model
        self.memory_model = memory_model
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

    def _create_memory_model_inputs(self, rollout):
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
        if step == 0:
            self.encoded_obs.append(enc_obs)
            # TODO(mmatena): Handle this case better.
            return self.learned_policy.sample_action(
                np.zeros([enc_obs.shape[0], self.learned_policy.in_size()])
            )
        inputs, mask, nonpadding_seqlen = self._create_memory_model_inputs(rollout)
        hidden_state = self.memory_model.get_hidden_representation(
            inputs, mask=mask, position=nonpadding_seqlen - 1
        )
        self.encoded_obs.append(enc_obs)
        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)
        action = self.learned_policy.sample_action(policy_input)
        return action
