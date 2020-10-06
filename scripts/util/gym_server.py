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

import ray

from pyvirtualdisplay import Display
import rpyc
from rpyc.utils.server import ThreadedServer
from rpyc.core.channel import Channel

from unittest import mock

from rl755.common import misc

# from rl755.environments.car_racing import CarRacing

import pyglet

# Errors otherwise in the singularity environment.
pyglet.options["debug_gl"] = False
if True:
    from pyglet import gl


Channel.COMPRESSION_LEVEL = 6

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 18861, "The port to listen on.")
flags.DEFINE_integer("parallelism", 1, "")

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
        self.envs = [gym.make(self.env_name) for _ in range(self.num_environments)]
        # self.envs = [CarRacing() for _ in range(self.num_environments)]
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


# class OpenAiGymService(rpyc.Service):
#     """Note that a new intance will be created for each connection."""

#     def __init__(self):
#         super().__init__()
#         self.env = None
#         self.parallelism = FLAGS.parallelism

#     def exposed_reset(self):
#         if self.env:
#             self.env.reset()

#     def exposed_make(self, env_name, num_environments):
#         self.env = GymEnvironments(
#             num_environments=num_environments,
#             env_name=env_name,
#         )

#     def exposed_render(self, whether_to_renders):
#         whether_to_renders = pickle.loads(whether_to_renders)
#         ret = self.env.render(whether_to_renders)
#         return pickle.dumps(ret)

#     def exposed_step(self, actions):
#         actions = pickle.loads(actions)
#         start = time.time()
#         ret = self.env.step(actions)
#         logging.info(f"Step time: {time.time() - start}")
#         return pickle.dumps(ret)

#     def exposed_close(self):
#         self.env.close()

RemoteGymEnvironments = ray.remote(GymEnvironments)


class OpenAiGymService(rpyc.Service):
    """Note that a new intance will be created for each connection."""

    def __init__(self):
        super().__init__()
        self.envs = []
        self.env = None
        self.parallelism = FLAGS.parallelism

    def exposed_reset(self):
        ray.get([env.reset.remote() for env in self.envs])

    def exposed_make(self, env_name, num_environments):
        partitions = misc.evenly_partition(num_environments, self.parallelism)
        self.envs = []
        for num_envs in partitions:
            env = RemoteGymEnvironments.remote(
                num_environments=num_envs,
                env_name=env_name,
            )
            self.envs.append(env)

    def exposed_render(self, whether_to_renders):
        whether_to_renders = pickle.loads(whether_to_renders)
        partitions = misc.evenly_partition(whether_to_renders, self.parallelism)
        ret = []
        for p, env in zip(partitions, self.envs):
            ret.append(env.render.remote(p))
        ret = ray.get(ret)
        # TODO: Need to combine stuff here.
        return pickle.dumps(ret)

    def exposed_step(self, actions):
        actions = pickle.loads(actions)
        partitions = misc.evenly_partition(actions, self.parallelism)
        ret = []
        for p, env in zip(partitions, self.envs):
            ret.append(env.step.remote(p))
        ret = ray.get(ret)
        # TODO: Need to combine stuff here.
        return pickle.dumps(ret)

    def exposed_close(self):
        ray.get([env.close.remote() for env in self.envs])


def main(_):
    display = Display(visible=0, size=(400, 300))
    display.start()
    ray.init()

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

    #######################################################
    # Using gpu partition, 8g memory, 12 cpu, 1 gpu
    # Times are for 10 steps with 128 envs.

    # No ray, fast car:
    #   2.7282283306121826

    # Ray, 1 parallelism, slow car:
    #   20.5257728099823
    # Ray, 2 parallelism, slow car:
    #   10.191755294799805
    # Ray, 4 parallelism, slow car:
    #   5.053348541259766
    # Ray, 8 parallelism, slow car:
    #   3.4551267623901367
    # Ray, 12 parallelism, slow car:
    #   2.750385046005249
    # Ray, 16 parallelism, slow car:
    #   2.5808982849121094
    # Ray, 32 parallelism, slow car:
    #   2.1441810131073
    # Ray, 64 parallelism, slow car:
    #   2.388378858566284

    # Ray, 1 parallelism, fast car:
    #   3.21494460105896
    # Ray, 2 parallelism, fast car:
    #   1.6657295227050781
    # Ray, 4 parallelism, fast car:
    #   1.0739614963531494
    # Ray, 8 parallelism, fast car:
    #   0.9639308452606201
    # Ray, 12 parallelism, fast car:
    #   0.8984279632568359
    # Ray, 16 parallelism, fast car:
    #   0.950084924697876

    #######################################################
    # Using gpu partition, 8g memory, 12 cpu, 4 gpu
    # Times are for 10 steps with 128 envs.

    # Ray, 1 parallelism, fast car:
    #   0

    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################
    # display = Display(visible=0, size=(400, 300))
    # display.start()

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
