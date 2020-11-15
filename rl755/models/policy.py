"""Policies.

Note that these don't contain any learnable parameters themselves.
Instead they let us interface components with the gym3 environment.
"""
import gym3
import numpy as np
import tensorflow as tf

from rl755.models.memory.interface import MemoryComponentWithHistory

ACTION_SIZE = 15


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
        assert not isinstance(
            memory_model, MemoryComponentWithHistory
        ), "Use PolicyWrapperWithHistory instead."
        self.vision_model = vision_model
        self.memory_model = memory_model
        self.learned_policy = learned_policy
        self.max_seqlen = max_seqlen

    def initialize(self, env, max_steps):
        self.encoded_obs = np.empty(
            [env.num, max_steps, self.vision_model.get_representation_size()],
            dtype=np.float32,
        )

    def get_window_of_preceding_observations(self, step):
        start_index = max(0, step - self.max_seqlen)
        return self.encoded_obs[:, start_index:step]

    def _ensure_sequence_length(self, x, sequence_length):
        x = x[..., -sequence_length:, :]
        diff = sequence_length - x.shape[-2]
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

    def _create_memory_model_inputs(self, rollout_state):
        observations = self.get_window_of_preceding_observations(rollout_state.step)
        actions = rollout_state.get_window_of_preceding_actions(self.max_seqlen)
        nonpadding_seqlen = observations.shape[1]

        actions = tf.one_hot(actions.T, depth=ACTION_SIZE, axis=-1)
        inputs = tf.concat([observations, actions], axis=-1)
        inputs, mask = self._ensure_sequence_length(inputs, self.max_seqlen)
        return inputs, mask, nonpadding_seqlen

    def sample_action(self, rollout_state):
        step = rollout_state.step
        obs = rollout_state.get_current_observation()
        obs = tf.cast(obs, tf.float32) / 255.0
        enc_obs = self.vision_model.compute_tensor_representation(obs, training=False)
        if step == 0:
            self.encoded_obs[:, step] = enc_obs
            # TODO(mmatena): Handle this case better.
            return self.learned_policy.sample_action(
                np.zeros([enc_obs.shape[0], self.learned_policy.in_size()])
            )
        inputs, mask, nonpadding_seqlen = self._create_memory_model_inputs(
            rollout_state
        )
        hidden_state = self.memory_model.get_hidden_representation(
            inputs,
            mask=mask,
            training=tf.constant(False),
            position=tf.constant(nonpadding_seqlen - 1),
        )
        self.encoded_obs[:, step] = enc_obs
        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)
        action = self.learned_policy.sample_action(policy_input)
        return action


class PolicyWrapperWithHistory(Policy):
    # Note: TODO: Add docs
    def __init__(self, vision_model, memory_model, learned_policy, max_seqlen):
        assert isinstance(
            memory_model, MemoryComponentWithHistory
        ), "Use PolicyWrapper instead."
        self.vision_model = vision_model
        self.memory_model = memory_model
        self.learned_policy = learned_policy
        self.max_seqlen = max_seqlen

    def initialize(self, env, max_steps):
        stride = self.memory_model.get_value_stride()
        max_history_values = 1 + (max_steps - self.max_seqlen) // stride

        self.history_values = np.zeros(
            [
                env.num,
                max_steps,
                self.vision_model.get_representation_size() + ACTION_SIZE,
            ],
            dtype=np.float32,
        )
        self.history_keys = np.zeros(
            [env.num, max_history_values, self.memory_model.get_key_size()],
            dtype=np.float32,
        )
        self.num_keys = 0

    def _create_memory_model_inputs(self, rollout_state):
        start_index = max(0, rollout_state.step - self.max_seqlen)
        end_index = min(self.max_seqlen, rollout_state.step)

        inputs = self.history_values[:, start_index : start_index + self.max_seqlen]
        nonpadding_seqlen = min(self.max_seqlen, end_index)

        return tf.constant(inputs), nonpadding_seqlen

    def _create_history_inputs(self, rollout_state):
        step = rollout_state.step
        num_envs = rollout_state.num_envs
        # We don't really care about this when training controllers since we ignore
        # all steps after the first episode terminates when computing the cumulative
        # reward.
        history_length = step * tf.ones([num_envs, 1], dtype=tf.int32)
        history = tf.constant(self.history_values)
        keys = tf.constant(self.history_keys)
        return history, keys, history_length, tf.constant(self.num_keys, dtype=tf.int32)

    def _possibly_add_key(self, step):
        stride = self.memory_model.get_value_stride()
        next_value_end = self.num_keys * stride + self.max_seqlen
        if step + 1 < next_value_end:
            return

        value = self.history_values[
            :, next_value_end - self.max_seqlen : next_value_end
        ]
        value = tf.constant(value)
        key = self.memory_model.key_from_value(value, training=tf.constant(False))

        self.history_keys[:, self.num_keys, :] = key
        self.num_keys += 1

    def _add_to_history(self, enc_obs, action, step):
        vision_size = self.vision_model.get_representation_size()
        self.history_values[:, step, :vision_size] = enc_obs
        self.history_values[:, step, vision_size + action] = 1.0
        self._possibly_add_key(step)

    def sample_action(self, rollout_state):
        step = rollout_state.step

        obs = rollout_state.get_current_observation()
        obs = tf.cast(obs, tf.float32) / 255.0

        enc_obs = self.vision_model.compute_tensor_representation(obs, training=False)
        if step == 0:
            # TODO(mmatena): Handle this case better.
            action = self.learned_policy.sample_action(
                np.zeros([enc_obs.shape[0], self.learned_policy.in_size()])
            )
            self._add_to_history(enc_obs, action, step)
            return action

        inputs, nonpadding_seqlen = self._create_memory_model_inputs(rollout_state)

        history, keys, history_length, num_keys = self._create_history_inputs(
            rollout_state
        )
        hidden_state = self.memory_model.get_hidden_representation(
            inputs,
            history=history,
            keys=keys,
            history_length=history_length,
            num_keys=num_keys,
            # Mask is not needed due to the autoregressive nature of the transformer.
            mask=None,
            training=tf.constant(False),
            position=tf.constant(nonpadding_seqlen - 1),
        )

        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)
        action = self.learned_policy.sample_action(policy_input)

        self._add_to_history(enc_obs, action, step)

        return action
