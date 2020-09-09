"""Learns a simple policy using CMA."""
import cma

from absl import app
from absl import flags

import numpy as np
from pyvirtualdisplay import Display
import rpyc
import tensorflow as tf

from rl755.data_gen import gym_rollouts
from rl755.models.car_racing import policies
from rl755.models.car_racing import saved_models


def fn(x):
    return np.sum((x - 5.0) ** 2)


class LinearPolicy(object):
    @staticmethod
    def from_flat_array(array, in_size, out_size):
        array = np.array(array)
        w, b = array[:-out_size], array[-out_size:]
        w = np.reshape(w, [out_size, in_size])
        return LinearPolicy(w=w, b=b)

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def sample_action(self, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        action = np.matmul(self.w, inputs) + self.b
        action = np.reshape(action, [3])
        return tuple(action.tolist())


# TODO(mmatena): Make this cleaner.
IP_FILE = "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt"
with open(IP_FILE, "r") as f:
    ip = f.read()

# TODO(mmatena): Make this configurable.
gym_service = rpyc.connect(ip, 18861, config={"allow_all_attrs": True}).root

encoder = saved_models.raw_rollout_vae_32ld()
sequence_model = saved_models.encoded_rollout_transformer()
sequence_model.return_layer_outputs = True
in_size = 256 + 32
out_size = 3
max_seqlen = 32


def get_score(flat_array, num_trials):
    linear_policy = LinearPolicy.from_flat_array(
        flat_array, in_size=in_size, out_size=out_size
    )
    policy = policies.CarRacingPolicy(
        encoder=encoder,
        sequence_model=sequence_model,
        policy=linear_policy,
        max_seqlen=max_seqlen,
    )
    # return gym_service.get_score(
    #     gym_service,
    #     num_trials=num_trials,
    #     initialize=policy.initialize,
    #     sample_action=policy.sample_action,
    # )
    rollouts = []
    gym_service.make("CarRacing-v0")
    gym_rollouts.serial_rollouts(
        gym_service,
        policy=policy,
        max_steps=2000,
        num_rollouts=num_trials,
        process_rollout_fn=lambda r: rollouts.append(r),
    )
    return np.mean([sum(r.reward_l) for r in rollouts])


# It looks like OpenAI gym requires some sort of display, so we
# have to fake one.
display = Display(visible=0, size=(400, 300))
display.start()

# es = cma.CMAEvolutionStrategy(8 * [0], 0.5, {"popsize": 64})
es = cma.CMAEvolutionStrategy(
    (in_size * out_size + out_size) * [0], 0.5, {"popsize": 2}
)
for i in range(2):
    solutions = es.ask()
    fitlist = np.zeros(es.popsize)

    for i in range(es.popsize):
        fitlist[i] = get_score(solutions[i], num_trials=2)

    es.tell(solutions, fitlist)
print(es.result)

# def main(_):
#     es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
#     for i in range(10):
#         solutions = es.ask()
#         fitlist = np.zeros(es.popsize)

#         for i in range(es.popsize):
#             fitlist[i] = fn(solutions[i])

#         es.tell(fitlist)
#         bestsol, bestfit = es.result()


# if __name__ == "__main__":
#     app.run(main)

# help(cma.fmin)
# help(cma.CMAEvolutionStrategy)
# help(cma.CMAOptions)
