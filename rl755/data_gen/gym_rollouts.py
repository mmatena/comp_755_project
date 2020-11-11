"""Script for generating many rollouts of a given policy."""
import numpy as np
from procgen import ProcgenGym3Env
import tensorflow as tf

from rl755.common import tfrecords


class RolloutState(object):
    # TODO: Add docs

    def __init__(self, num_envs, max_steps, obs_shape=(64, 64, 3)):
        # TODO: Add docs
        obs_shape = list(obs_shape)
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.obs_shape = obs_shape

        self.step = 0

        # TODO: Document these (and done_steps):
        self.observations = np.empty(
            [max_steps + 1, num_envs] + obs_shape, dtype=np.uint8
        )
        self.rewards = np.empty([max_steps, num_envs], dtype=np.float32)
        self.actions = np.empty([max_steps, num_envs], dtype=np.uint8)

        # observation[:ds+1], rewards[:ds], actions[:ds] are the first rollout.
        self.done_steps = max_steps * np.ones([num_envs], dtype=np.int32)

    def set_initial_observation(self, env):
        assert self.step == 0, "Must be at step 0 to set the initial observation."
        _, obs, _ = env.observe()
        self.observations[0] = obs["rgb"]

    def get_current_observation(self):
        return self.observations[self.step]

    def get_window_of_preceding_actions(self, size):
        start_index = max(0, self.step - size)
        return self.actions[start_index : self.step]

    def perform_step(self, env, action):
        env.act(action)
        rew, obs, firsts = env.observe()

        self.observations[self.step + 1] = obs["rgb"]
        self.rewards[self.step] = rew
        self.actions[self.step] = action

        self.step += 1

        first_dones = np.logical_and(self.done_steps >= self.step, firsts)
        self.done_steps[first_dones] = self.step

    def to_serialized_tfrecords(self):
        # TODO: Add docs
        records = []
        for i in range(self.num_envs):
            feature_list = {
                "observations": tfrecords.to_bytes_feature_list(
                    self.observations[:, i]
                ),
                "rewards": tfrecords.to_float_feature_list(self.rewards[:, i][:, None]),
                "actions": tfrecords.to_int64_feature_list(self.actions[:, i][:, None]),
                "done_step": tfrecords.to_int64_feature_list([[self.done_steps[i]]]),
            }
            example_proto = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list=feature_list)
            )
            records.append(example_proto.SerializeToString())
        return records

    def get_first_rollout_total_reward(self):
        # TODO: Add docs
        mask = np.arange(self.max_steps)[:, None] < self.done_steps[None, :]
        mask = mask.astype(np.int32)
        return np.sum(self.rewards * mask, axis=0)


def perform_rollouts(env_name, num_envs, policy, max_steps, **env_kwargs):
    # TODO(mmatena): Add docs
    # TODO(mmatena): Other options
    if "start_level" not in env_kwargs:
        env_kwargs["start_level"] = 0
    if "num_levels" not in env_kwargs:
        # Use unlimited levels.
        env_kwargs["num_levels"] = 0
    if "distribution_mode" not in env_kwargs:
        env_kwargs["distribution_mode"] = "easy"

    env = ProcgenGym3Env(env_name=env_name, num=num_envs, **env_kwargs)

    rollout_state = RolloutState(num_envs=num_envs, max_steps=max_steps)
    rollout_state.set_initial_observation(env)

    policy.initialize(env, max_steps=max_steps)

    for step in range(max_steps):
        import time

        start = time.time()
        action = policy.sample_action(rollout_state)
        print(time.time() - start)
        rollout_state.perform_step(env, action)

    env.close()

    return rollout_state
