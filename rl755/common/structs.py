"""Common structs."""
from collections import namedtuple

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
