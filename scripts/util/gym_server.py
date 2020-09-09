from absl import app
from absl import flags

import numpy as np
from pyvirtualdisplay import Display
import rpyc
from rpyc.utils.server import ThreadedServer

from rl755.data_gen import gym_rollouts

FLAGS = flags.FLAGS

flags.DEFINE_integer("port", 18861, "The port to listen on.")


class OpenAiGymService(rpyc.Service):
    """Note that a new intance will be created for each connection."""

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_get_score(env_name, policy, num_trials):
        rollouts = []
        gym_rollouts.serial_rollouts(
            env_name,
            policy=policy,
            max_steps=2000,
            num_rollouts=num_trials,
            process_rollout_fn=lambda r: rollouts.append(r),
        )
        return np.mean([sum(r.reward_l) for r in rollouts])


def main(_):
    # It looks like OpenAI gym requires some sort of display, so we
    # have to fake one.
    display = Display(visible=0, size=(400, 300))
    display.start()

    t = ThreadedServer(OpenAiGymService, port=FLAGS.port)
    t.start()


if __name__ == "__main__":
    app.run(main)
