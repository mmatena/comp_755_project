import time
import pickle

from absl import app
from absl import flags

import gym
import numpy as np
from pyvirtualdisplay import Display
import rpyc
from rpyc.utils.server import ThreadedServer

from rl755.data_gen import gym_rollouts

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 18861, "The port to listen on.")

IP_FILE = "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt"


def image_to_tuples(image):
    return tuple(tuple(tuple(b for b in a)) for a in image)


class OpenAiGymService(rpyc.Service):
    """Note that a new intance will be created for each connection."""

    def __init__(self):
        super().__init__()
        self.env = None

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

    def exposed_make(self, env_name):
        self.env = gym.make(env_name)

    def exposed_render(self, *args, **kwargs):
        start = time.time()
        image = self.env.render(*args, **kwargs)
        print(f"Render time: {time.time() - start} s")
        return pickle.dumps(image)

    def exposed_step(self, action):
        start = time.time()
        _, reward, done, _ = self.env.step(action)
        print(f"Step time: {time.time() - start} s")
        return None, reward, done, None

    def exposed_close(self):
        if self.env:
            self.env.close()


def main(_):
    import socket

    hostname = socket.gethostbyname(socket.gethostname())
    with open(IP_FILE, "w+") as f:
        f.write(hostname)

    # It looks like OpenAI gym requires some sort of display, so we
    # have to fake one.
    display = Display(visible=0, size=(400, 300))
    display.start()

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
