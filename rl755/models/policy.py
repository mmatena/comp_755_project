"""Policies.

Note that these don't contain any learnable parameters themselves.
Instead they let us interface components with the gym3 environment.
"""
import gym3
import numpy as np
import tensorflow as tf

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
        self.vision_model = vision_model
        self.memory_model = memory_model
        self.learned_policy = learned_policy
        self.max_seqlen = max_seqlen

        print("TODO: Make sure everything in this class lines up!!!")

    def initialize(self, env, max_steps):
        # TODO: Probably make this into a np array, need to change parts that use it.
        self.encoded_obs = np.empty(
            [env.num, max_steps, self.vision_model.get_representation_size()],
            dtype=np.float32,
        )

    def get_window_of_preceding_observations(self, step):
        start_index = max(0, step - self.max_seqlen)
        return self.encoded_obs[:, start_index:step]

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

    def _create_memory_model_inputs(self, rollout_state):
        observations = self.get_window_of_preceding_observations(rollout_state.step)
        actions = rollout_state.get_window_of_preceding_actions(self.max_seqlen)
        nonpadding_seqlen = observations.shape[1]

        actions = tf.one_hot(actions.T, depth=ACTION_SIZE, axis=-1)
        inputs = tf.concat([observations, actions], axis=-1)
        inputs, mask = self._ensure_sequence_length(inputs)
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
            inputs, mask=mask, training=False, position=nonpadding_seqlen - 1
        )
        self.encoded_obs[:, step] = enc_obs
        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)
        action = self.learned_policy.sample_action(policy_input)
        return action
