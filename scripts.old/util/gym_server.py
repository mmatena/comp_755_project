import collections
import enum
import os
import pickle
import socket
import time
import functools

from absl import logging

from absl import app
from absl import flags

import gym

import numpy as np

from pyvirtualdisplay import Display
import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.core.channel import Channel

from unittest import mock

from rl755.common import misc
from rl755.common.structs import StepInfo

from rl755.environments import Environments
from rl755.environments.car_racing import CarRacing

import pyglet

# Errors otherwise in the singularity environment.
pyglet.options["debug_gl"] = False
if True:
    from pyglet import gl


Channel.COMPRESSION_LEVEL = 6

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 18861, "The port to listen on.")
flags.DEFINE_string(
    "gym_server_ip_file",
    "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt",
    "We write the ip address of our gym server here so that other programs can "
    "read it and find our server.",
)


class GymEnvironments(object):
    """Multiple synchronous gym environments."""

    def __init__(self, num_environments, env_name):
        self.num_environments = num_environments
        self.env_name = env_name
        self.env_enum = Environments.environment_from_open_ai_name(env_name)
        self._create_envs()

    def _create_envs(self):
        if self.env_name == "CarRacing-v0":
            # This version is significantly faster.
            self.envs = [CarRacing() for _ in range(self.num_environments)]
        else:
            self.envs = [gym.make(self.env_name) for _ in range(self.num_environments)]
        for env in self.envs:
            env.reset()

    def close(self):
        for env in self.envs:
            env.close()

    def step(self, actions):
        assert len(actions) == len(self.envs)
        observations = np.empty(
            [self.num_environments] + list(self.env_enum.observation_shape),
            dtype=np.uint8,
        )
        rewards = np.empty([self.num_environments], dtype=np.float32)
        for i, (action, env) in enumerate(zip(actions, self.envs)):
            if not self.dones[i]:
                obs, reward, done, _ = env.step(action)
                observations[i] = obs
                rewards[i] = reward
                self.dones[i] = done
        return StepInfo(reward=rewards, done=self.dones, observation=observations)

    def render(self):
        observations = np.empty(
            [self.num_environments] + list(self.env_enum.observation_shape),
            dtype=np.uint8,
        )
        for i, env in enumerate(self.envs):
            observations[i] = env.render("state_pixels")
        return observations

    def reset(self):
        self.dones = np.zeros([self.num_environments], dtype=np.bool)
        for env in self.envs:
            env.reset()


class OpenAiGymService(rpyc.Service):
    """Note that a new intance will be created for each connection."""

    def __init__(self):
        super().__init__()
        self.env = None

    def exposed_reset(self):
        if self.env:
            self.env.reset()

    def exposed_make(self, env_name, num_environments):
        self.env = GymEnvironments(
            num_environments=num_environments,
            env_name=env_name,
        )

    def exposed_render(self):
        ret = self.env.render()
        return pickle.dumps(ret)

    def exposed_step(self, actions):
        actions = pickle.loads(actions)
        ret = self.env.step(actions)
        return pickle.dumps(ret)

    def exposed_close(self):
        self.env.close()


def main(_):
    display = Display(visible=0, size=(400, 300))
    display.start()

    hostname = socket.gethostbyname(socket.gethostname())
    with open(FLAGS.gym_server_ip_file, "w+") as f:
        f.write(hostname)

    t = ThreadedServer(
        OpenAiGymService,
        port=FLAGS.port,
        protocol_config={
            "allow_pickle": True,
        },
    )
    t.start()


if __name__ == "__main__":
    app.run(main)
