"""Policies for running in the OpenAI Gym.


Here is the information about a 3-d action vector:
    a[0]: [-1, 1], steering
    a[1]: [0, 1], gas
    a[2]: [0, 1], brakes
Values below/above the range are clipped to the min/max of the range, respectively.

"""
import tensorflow as tf
import tensorflow_probability as tfp

from rl755.data_gen import gym_rollouts

tfd = tfp.distributions


# a[0]: [-1, 1], steering
# a[1]: [0, 1], gas
# a[2]: [0, 1], brakes
# a[3]: unused


class CarRacingPolicy(gym_rollouts.Policy):
    def __init__(self, encoder, sequence_model, policy, max_seqlen=32):
        self.encoder = encoder
        self.sequence_model = sequence_model
        self.policy = policy
        self.max_seqlen = max_seqlen

    def initialize(self, env, max_steps, **kwargs):
        self.encoded_obs = []

    def _broadcast_across_samples(self, x):
        x = tf.expand_dims(x, axis=0)
        x = tf.broadcast_to(x, [self.num_samples] + x.shape[1:])
        return x

    def _ensure_sequence_length(self, x):
        x = x[:, -self.max_seqlen :]
        diff = self.max_seqlen - x.shape[1]
        if diff:
            mask = tf.concat(
                [tf.ones(x.shape[:2]), tf.zeros([x.shape[0], diff])], axis=1
            )
            padding = tf.zeros([x.shape[0], diff, x.shape[2]], dtype=tf.float32)
            x = tf.concat([x, padding], axis=1)
            return x, mask
        else:
            return x, None

    def _create_inputs(self, rollout):
        # o[i], a[i],] => o[i+1] or o[i+1] - o[i]
        observations = tf.concat(self.encoded_obs[-self.max_seqlen :], axis=0)
        observations = self._broadcast_across_samples(observations)

        actions = tf.constant(rollout.actions[-self.max_seqlen :], dtype=tf.float32)
        actions = self._broadcast_across_samples(actions)

        inputs = tf.concat([observations, actions], axis=-1)
        return self._ensure_sequence_length(inputs)

    def sample_action(self, obs, step, rollout, **kwargs):
        inputs, mask = self._create_inputs(rollout)
        # TODO(mmatena): This could be potentially be made hugely more efficient by reusing computations.
        hidden_state = self.sequence_model.get_last_representation_tensor(
            inputs, mask=mask
        )

        enc_obs = self.encoder.encode_tensor(
            tf.expand_dims(obs, axis=0), training=False
        )
        self.encoded_obs.append(enc_obs)

        policy_input = tf.concat([enc_obs, hidden_state], axis=-1)

        action = self.policy.sample_action(policy_input)

        return action


class RandomShootingPolicy(gym_rollouts.Policy):
    # TODO(mmatena): Probably deprecated.
    def __init__(
        self, encoder, model, num_samples=128, max_seqlen=32, action_dist=None
    ):
        # TODO(mmatena): Add docs
        self.encoder = encoder
        self.model = model
        self.num_samples = num_samples
        if action_dist is None:
            action_dist = tfd.Uniform(low=[-1, 0, 0], high=[1, 1, 1])
        self.action_dist = action_dist
        self.seqlen = max_seqlen

        self.encoded_obs = []

    def initialize(self, env, max_steps, **kwargs):
        self.encoded_obs = []

    def _broadcast_across_samples(self, x):
        x = tf.expand_dims(x, axis=0)
        x = tf.broadcast_to(x, [self.num_samples] + x.shape[1:])
        return x

    def _ensure_sequence_length(self, x):
        x = x[:, -self.seqlen :]
        diff = self.seqlen - x.shape[1]
        if diff:
            mask = tf.concat(
                [tf.ones(x.shape[:2]), tf.zeros([x.shape[0], diff])], axis=1
            )
            padding = tf.zeros([x.shape[0], diff, x.shape[2]], dtype=tf.float32)
            x = tf.concat([x, padding], axis=1)
            return x, mask
        else:
            return x, None

    def _create_inputs(self, test_actions, rollout):
        # o[i], a[i], r[i-1] => o[i+1], r[i]
        observations = tf.concat(self.encoded_obs[-self.seqlen :], axis=0)
        observations = self._broadcast_across_samples(observations)

        previous_actions = tf.constant(
            rollout.actions[-self.seqlen + 1 :], dtype=tf.float32
        )
        previous_actions = self._broadcast_across_samples(previous_actions)

        actions = tf.concat(
            [previous_actions, tf.expand_dims(test_actions, axis=1)], axis=1
        )

        rewards = tf.constant(rollout.rewards[-self.seqlen + 1 :], dtype=tf.float32)
        rewards = tf.concat([[0.0], rewards], axis=0)
        rewards = tf.expand_dims(rewards, axis=-1)
        rewards = self._broadcast_across_samples(rewards)

        inputs = tf.concat([observations, actions, rewards], axis=-1)

        return self._ensure_sequence_length(inputs)

    def sample_action(self, obs, step, rollout, **kwargs):
        print(f"Step {step}")
        actions = self.action_dist.sample(self.num_samples)
        # TODO(mmatena): Figure out the best way to deal with this unused action.
        actions = tf.concat([actions, tf.zeros([self.num_samples, 1])], axis=-1)

        enc_obs = self.encoder(tf.expand_dims(obs, axis=0), training=False)
        self.encoded_obs.append(enc_obs)

        # TODO(mmatena): This could be potentially be made hugely more efficient by reusing computations.
        inputs, mask = self._create_inputs(actions, rollout)

        outputs = self.model(inputs, mask=mask, training=False)
        if mask:
            predicted_reward = outputs[:, len(self.encoded_obs) - 1, -1]
        else:
            predicted_reward = outputs[:, -1, -1]
        best_action_index = tf.argmax(predicted_reward).numpy()
        return actions[best_action_index]
