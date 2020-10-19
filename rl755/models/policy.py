"""Policies.

Note that these don't contain any learnable parameters themselves.
Instead they let us interface components with the gym3 environment.
"""
import gym3


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
