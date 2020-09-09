"""Script for generating many rollouts of a given policy."""
import gym
from noise import pnoise1
import numpy as np
import ray

from rl755.common.misc import evenly_partition
from rl755.common.structs import Rollout


class Policy(object):
    """Abstract base class for policies"""

    def initialize(self, env, max_steps, **kwargs):
        """Initializes the policy upon starting a new episode.

        Please add a (potentially) unused **kwargs to subclasses to prevent breakage
        if we need to expose more information for some policies in the future.

        Args:
            env: the OpenAI Gym environment we will be running on
            max_steps: a postive integer, the maximum number of steps the
                rollout will go on for
        """
        raise NotImplementedError()

    def sample_action(self, obs, step, rollout, **kwargs):
        """Generate an action.

        Please add a (potentially) unused **kwargs to subclasses to prevent breakage
        if we need to expose more information for some policies in the future.

        Args:
            obs: the observation at the current time step, whose exact format depends
                on the environment you are running on
            step: non-negative integer, the 0-based index of the current time step
            rollout: the Rollout object corresponding to this run
        Returns:
            A action compatible with the `env.step(action)` method.
        """
        raise NotImplementedError()


class HastingsRandomPolicy(Policy):
    """The random policy that Hastings Greer was using.

    Currently only supports the CarRacing-v0 environment.
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


def single_rollout(env, policy, max_steps):
    """Runs a single rollout.

    The rollout will end when either the environment says we are done or we have
    executed for `max_steps`.

    Args:
        env: an OpenAI Gym environment
        policy: a subclass of rl755.data_gen.gym_rollouts.Policy
        max_steps: a positive integer, the maximum number of steps the rollout
            will go on for
    Returns:
        A rl755.common.structs.Rollout instance.
    """
    env.reset()
    policy.initialize(env=env, max_steps=max_steps)

    rollout = Rollout()
    for step in range(max_steps):
        # TODO(mmatena): Support environments without a "state_pixels" render mode.
        obs = env.render("state_pixels")
        action = policy.sample_action(obs=obs, step=step, rollout=rollout)
        _, reward, done, _ = env.step(action)

        rollout.obs_l.append(obs)
        rollout.action_l.append(action)
        rollout.reward_l.append(reward)

        if done:
            rollout.done = True
            break

    return rollout


def serial_rollouts(env_name, policy, max_steps, num_rollouts, process_rollout_fn):
    """Runs `num_rollouts` and applies `process_rollout_fn` to each generated Rollout object."""
    if isinstance(env_name, str):
        env = gym.make(env_name)
    else:
        env = env_name
    for _ in range(num_rollouts):
        rollout = single_rollout(env, policy=policy, max_steps=max_steps)
        process_rollout_fn(rollout)
    env.close()


def parallel_rollouts(
    env_name, policy, max_steps, num_rollouts, process_rollout_fn, parallelism=1
):
    """Runs `num_rollouts` with up to `parallelism` rollouts occuring concurrently."""
    fn = ray.remote(serial_rollouts)
    do_rollouts = lambda num_serial: fn.remote(
        env_name=env_name,
        policy=policy,
        max_steps=max_steps,
        num_rollouts=num_serial,
        process_rollout_fn=process_rollout_fn,
    )
    futures = [do_rollouts(num) for num in evenly_partition(num_rollouts, parallelism)]
    return ray.get(futures)
