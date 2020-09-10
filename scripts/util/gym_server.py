import collections
import enum
import multiprocessing
import pickle
import time

from absl import app
from absl import flags

# import gym
import numpy as np

# from pyvirtualdisplay import Display
import rpyc
from rpyc.utils.server import ThreadedServer

from rl755.common import misc
from rl755.data_gen import gym_rollouts

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "processes", None, "The number of processes to use for each connection."
)
flags.DEFINE_integer("port", 18861, "The port to listen on.")

flags.mark_flag_as_required("processes")

IP_FILE = "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt"


# Information to be returned after we do a step.
StepInfo = collections.namedtuple("StepInfo", ["reward", "done"])

# Structure used to pass data into a GymEnvironments process.
InMessage = collections.namedtuple("InMessage", ["type", "data"])

OutMessage = collections.namedtuple("OutMessage", ["index", "data"])


class MessageType(enum.Enum):
    KILL = 1
    STEP = 2
    RENDER = 3
    RESET = 4


if True:
    import sys
    import pdb


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class GymEnvironments(multiprocessing.Process):
    """A single process that runs multiple gym environments."""

    def __init__(
        self, index, num_environments, env_name, in_queue, step_info_queue, render_queue
    ):
        super().__init__()
        from pyvirtualdisplay import Display

        self.display = Display(visible=0, size=(400, 300))
        self.display.start()

        self.index = index
        self.in_queue = in_queue
        self.step_info_queue = step_info_queue
        self.render_queue = render_queue
        self.num_environments = num_environments
        self.env_name = env_name

    def _create_envs(self):
        import gym
        import pyglet

        self.envs = [gym.make(self.env_name) for _ in range(self.num_environments)]
        for env in self.envs:
            env.reset()

    def _kill(self):
        for env in self.envs:
            env.close()

    def _step(self, actions):
        # actions.shape = [num_environments, action_dim]
        assert len(actions) == len(self.envs)
        ret = []
        for action, env in zip(actions, self.envs):
            # None as actions means do nothing, can we used when one env
            # is finished but others aren't.
            if action is None:
                ret.append(None)
                continue
            _, reward, done, _ = env.step(action)
            ret.append(StepInfo(reward=reward, done=done))
        self.step_info_queue.put(OutMessage(index=self.index, data=ret))

    def _render(self, whether_to_renders):
        assert len(whether_to_renders) == len(self.envs)
        ret = []
        for should_render, env in zip(whether_to_renders, self.envs):
            if should_render:
                ret.append(env.render("state_pixels"))
            else:
                ret.append(None)
        self.render_queue.put(OutMessage(index=self.index, data=ret))

    def _reset(self):
        for env in self.envs:
            env.reset()

    def run(self):
        self._create_envs()
        while True:
            msg = self.in_queue.get()
            if msg.type == MessageType.KILL:
                self._kill()
                return
            elif msg.type == MessageType.STEP:
                self._step(msg.data)
            elif msg.type == MessageType.RENDER:
                self._render(msg.data)
            elif msg.type == MessageType.RESET:
                self._reset()
            else:
                raise ValueError(f"Invalid message type: {msg.type}")


class OpenAiGymService(rpyc.Service):
    """Note that a new intance will be created for each connection."""

    def __init__(self):
        super().__init__()
        self.envs = None
        self.step_info_queue = multiprocessing.Queue()
        self.render_queue = multiprocessing.Queue()
        self.num_processes = FLAGS.processes

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_reset(self):
        if self.env:
            self.env.reset()

    def exposed_make(self, env_name, num_environments):
        partitions = misc.evenly_partition(num_environments, self.num_processes)
        self.envs = [
            GymEnvironments(
                index=index,
                num_environments=size,
                env_name=env_name,
                in_queue=multiprocessing.Queue(),
                step_info_queue=self.step_info_queue,
                render_queue=self.render_queue,
            )
            for index, size in enumerate(partitions)
        ]
        for env in self.envs:
            env.start()

    def exposed_render(self, whether_to_renders):
        whether_to_renders = pickle.loads(whether_to_renders)
        whether_to_renders = misc.evenly_partition(
            whether_to_renders, self.num_processes
        )
        assert len(whether_to_renders) == len(self.envs)
        for wtr, env in zip(whether_to_renders, self.envs):
            env.in_queue.put_nowait(InMessage(type=MessageType.RENDER, data=wtr))
        ret = self.num_processes * [None]

        for _ in self.envs:
            msg = self.render_queue.get()
            ret[msg.index] = msg.data
        return pickle.dumps(ret)

    def exposed_step(self, actions):
        actions = pickle.loads(actions)
        actions = misc.evenly_partition(actions, self.num_processes)
        assert len(actions) == len(self.envs)
        for action, env in zip(actions, self.envs):
            env.in_queue.put_nowait(InMessage(type=MessageType.STEP, data=action))
        ret = self.num_processes * [None]
        for _ in self.envs:
            msg = self.step_info_queue.get()
            ret[msg.index] = msg.data
        return pickle.dumps(ret)

    def exposed_close(self):
        raise ValueError("TODO")

    # def exposed_reset(self):
    #     if self.env:
    #         self.env.reset()

    # def exposed_make(self, env_name):
    #     self.env = gym.make(env_name)

    # def exposed_render(self, *args, **kwargs):
    #     image = self.env.render(*args, **kwargs)
    #     return pickle.dumps(image)

    # def exposed_step(self, action):
    #     _, reward, done, _ = self.env.step(action)
    #     return None, reward, done, None

    # def exposed_close(self):
    #     if self.env:
    #         self.env.close()


# It looks like OpenAI gym requires some sort of display, so we
# have to fake one.
# display = Display(visible=0, size=(400, 300))
# display.start()


def main(_):
    import socket

    hostname = socket.gethostbyname(socket.gethostname())
    with open(IP_FILE, "w+") as f:
        f.write(hostname)

    # # It looks like OpenAI gym requires some sort of display, so we
    # # have to fake one.
    # display = Display(visible=0, size=(400, 300))
    # display.start()

    FACTOR = 1
    s = OpenAiGymService()
    s.exposed_make("CarRacing-v0", FACTOR * FLAGS.processes)
    # s.exposed_render(pickle.dumps(FACTOR * FLAGS.processes * [True]))
    # s.exposed_render(pickle.dumps(FACTOR * FLAGS.processes * [True]))
    # s.exposed_render(pickle.dumps(FACTOR * FLAGS.processes * [True]))
    # s.exposed_step(pickle.dumps(FACTOR * FLAGS.processes * [[1, 1.0, 1]]))
    # s.exposed_step(pickle.dumps(FACTOR * FLAGS.processes * [[1, 1.0, 1]]))
    start = time.time()
    s.exposed_step(pickle.dumps(FACTOR * FLAGS.processes * [[1, 1.0, 1]]))
    print(time.time() - start)

    if True:
        return

    t = ThreadedServer(
        OpenAiGymService,
        port=FLAGS.port,
        protocol_config={
            "allow_public_attrs": True,
            "allow_all_attrs": True,
        },
    )
    t.start()


if __name__ == "__main__":
    app.run(main)
