"""Learns a simple policy using CMA."""
import functools
import multiprocessing
import time

from absl import app
from absl import flags
import cma
import numpy as np
from pyvirtualdisplay import Display
import rpyc
import tensorflow as tf

from rl755.common import misc
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
conn = rpyc.connect(ip, 18861, config={"allow_all_attrs": True})
conn._config["sync_request_timeout"] = None
gym_service = conn.root


encoder = saved_models.raw_rollout_vae_32ld()
sequence_model = saved_models.encoded_rollout_transformer()
sequence_model.return_layer_outputs = True
in_size = 256 + 32
out_size = 3
max_seqlen = 32


# def get_score(flat_array, num_trials):
#     linear_policy = LinearPolicy.from_flat_array(
#         flat_array, in_size=in_size, out_size=out_size
#     )
#     policy = policies.CarRacingPolicy(
#         encoder=encoder,
#         sequence_model=sequence_model,
#         policy=linear_policy,
#         max_seqlen=max_seqlen,
#     )
#     # return gym_service.get_score(
#     #     gym_service,
#     #     num_trials=num_trials,
#     #     initialize=policy.initialize,
#     #     sample_action=policy.sample_action,
#     # )
#     rollouts = []
#     gym_service.make("CarRacing-v0")
#     print("Increase MAX STEPS!!!!!")
#     gym_rollouts.serial_rollouts(
#         gym_service,
#         policy=policy,
#         # max_steps=2000,
#         max_steps=100,
#         num_rollouts=num_trials,
#         process_rollout_fn=lambda r: rollouts.append(r),
#     )
#     return np.mean([sum(r.reward_l) for r in rollouts])


def get_score(flat_array):
    # conn = rpyc.connect(ip, 18861, config={"allow_all_attrs": True})
    # conn._config["sync_request_timeout"] = None
    # gym_service = conn.root

    linear_policy = LinearPolicy.from_flat_array(
        flat_array, in_size=in_size, out_size=out_size
    )
    policy = policies.CarRacingPolicy(
        encoder=encoder,
        sequence_model=sequence_model,
        policy=linear_policy,
        max_seqlen=max_seqlen,
    )
    gym_service.make("CarRacing-v0")
    print("Increase MAX STEPS!!!!!")
    return gym_rollouts.single_rollout(gym_service, policy, max_steps=100)


# It looks like OpenAI gym requires some sort of display, so we
# have to fake one.
display = Display(visible=0, size=(400, 300))
display.start()

# ray.init(address="localhost:6379")

# es = cma.CMAEvolutionStrategy(8 * [0], 0.5, {"popsize": 64})
es = cma.CMAEvolutionStrategy(
    (in_size * out_size + out_size) * [0], 0.5, {"popsize": 2}
)
for i in range(2):
    start = time.time()
    solutions = es.ask()

    num_trials = 2
    args = functools.reduce(list.__add__, [num_trials * [s] for s in solutions])

    # fitlist = np.zeros(es.popsize)
    # for i in range(es.popsize):
    #     fitlist[i] = get_score(solutions[i], num_trials=2)
    processes = 6
    with multiprocessing.Pool(processes=processes) as pool:
        scores = pool.map(get_score, args)
    scores = misc.divide_chunks(scores, num_trials)
    fitlist = [sum(s) for s in scores]

    es.tell(solutions, fitlist)

    print(f"CMA step time: {time.time() - start} s")
# print(es.result)

# CMA step time: 21 + 27 s
