"""Script for generating many rollouts of a given policy."""
import gym
from noise import pnoise1
import numpy as np
import ray

from rl755.common.structs import Rollout


class Policy(object):
  """Base class for policies"""

  def initialize(self, env, max_steps, **kwargs):
    raise NotImplementedError()

  def sample_action(self, obs, step, **kwargs):
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
        action = [self._magnitude_scale * pnoise1(i / self._time_scale + start_i, 5) for start_i in start]
        out.append(action)
    self._actions = np.array(out)

  def sample_action(self, obs, step, **kwargs):
    del obs, kwargs
    return self._actions[step]


def single_rollout(env, policy, max_steps):
  env.reset()
  policy.initialize(env=env, max_steps=max_steps)

  rollout = Rollout()
  for step in range(max_steps):
    # TODO(mmatena): Support environments without a "state_pixels" render mode.
    obs = env.render("state_pixels")
    action = policy.sample_action(obs=obs, step=step)
    _, reward, done, _ = env.step(action)

    run.obs_l.append(obs)
    run.action_l.append(action)
    run.reward_l.append(reward)

    if done:
      rollout.done = True
      break

  return rollout


def serial_rollouts(env_name, policy, max_steps, num_rollouts, process_rollout_fn):
  env = gym.make(env_name)
  for _ in range(num_rollouts):
    rollout = single_rollout(env, policy=policy, max_steps=max_steps)
    process_rollout_fn(rollout)
  env.close()


def parallel_rollouts(env_name, policy, max_steps, num_rollouts, process_rollout_fn, parallelism=1):
  fn = ray.remote(serial_rollouts)
  do_rollouts = lambda num_serial: fn.remote(env_name=env_name,
                                             policy=policy,
                                             max_steps=max_steps,
                                             num_rollouts=num_serial,
                                             process_rollout_fn=process_rollout_fn)

  # TODO(mmatena): Better handling of the surplus. Spread evenly across the rest of the nodes instead.
  full_num_serial, surplus_num_serial = divmod(num_rollouts, parallelism)
  futures = [do_rollouts(full_num_serial) for _ in range(parallelism-1)]
  futures += [do_rollouts(full_num_serial + surplus_num_serial)]

  return ray.get(futures)

