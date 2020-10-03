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

Channel.COMPRESSION_LEVEL = 6

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 18861, "The port to listen on.")

IP_FILE = "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt"


# Information to be returned after we do a step.
StepInfo = collections.namedtuple("StepInfo", ["reward", "done", "observation"])


class GymEnvironments(object):
    """Multiple synchornized gym environments."""

    def __init__(self, num_environments, env_name):
        self.num_environments = num_environments
        self.env_name = env_name
        self._create_envs()

    def _create_envs(self):
        self.display = Display(visible=0, size=(400, 300))
        self.display.start()
        from gym.envs.classic_control import rendering

        _original_get_display = rendering.get_display

        def our_get_display(spec, actual_spec):
            del spec
            print("@@@@", actual_spec)
            return _original_get_display(actual_spec)

        with mock.patch.object(
            rendering,
            "get_display",
            functools.partial(our_get_display, actual_spec=f":{self.display.display}"),
        ):

            self.envs = [gym.make(self.env_name) for _ in range(self.num_environments)]
        for env in self.envs:
            env.reset()

    def close(self):
        for env in self.envs:
            env.close()
        self.display.close()

    def step(self, actions):
        assert len(actions) == len(self.envs)
        # TODO: support other shapes
        observations = np.empty([self.num_environments, 96, 96, 3], dtype=np.uint8)
        rewards = np.empty([self.num_environments], dtype=np.float32)
        for i, (action, env) in enumerate(zip(actions, self.envs)):
            obs, reward, done, _ = env.step(action)
            observations[i] = obs
            rewards[i] = reward
            # TODO(mmatena): Something with the dones
        return StepInfo(reward=rewards, done=None, observation=observations)

    def render(self, whether_to_renders):
        assert len(whether_to_renders) == len(self.envs)
        ret = []
        for should_render, env in zip(whether_to_renders, self.envs):
            if should_render:
                ret.append(env.render("state_pixels"))
            else:
                ret.append(None)
        return ret

    def reset(self):
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

    def exposed_render(self, whether_to_renders):
        whether_to_renders = pickle.loads(whether_to_renders)
        ret = self.env.render(whether_to_renders)
        return pickle.dumps(ret)

    def exposed_step(self, actions):
        actions = pickle.loads(actions)
        start = time.time()
        ret = self.env.step(actions)
        logging.info(f"Step time: {time.time() - start}")
        return pickle.dumps(ret)

    def exposed_close(self):
        self.env.close()


def main(_):
    BATCH = 128
    s = OpenAiGymService()
    s.exposed_make("CarRacing-v0", BATCH)
    s.exposed_step(pickle.dumps(BATCH * [[1, 1.0, 1]]))
    s.exposed_step(pickle.dumps(BATCH * [[1, 1.0, 1]]))

    start = time.time()
    for _ in range(5):
        s.exposed_step(pickle.dumps(BATCH * [[1, 1.0, 1]]))
        s.exposed_step(pickle.dumps(BATCH * [[1, 1.0, 1]]))
    logging.info(f"Time: {time.time() - start}")
    # 16.6 / 5
    # 33.3 / 10

    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################

    # hostname = socket.gethostbyname(socket.gethostname())
    # with open(IP_FILE, "w+") as f:
    #     f.write(hostname)

    # t = ThreadedServer(
    #     OpenAiGymService,
    #     port=FLAGS.port,
    #     protocol_config={
    #         "allow_pickle": True,
    #     },
    # )
    # t.start()


if __name__ == "__main__":
    app.run(main)
