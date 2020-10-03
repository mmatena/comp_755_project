"""Learns a simple policy using CMA."""
import collections
import functools
import multiprocessing
import pickle
import time
import zlib

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
from rl755.models.car_racing import transformer
from rl755.models.car_racing import saved_models
from rl755.common.structs import Rollout

StepInfo = collections.namedtuple("StepInfo", ["reward", "done", "observation"])


def fn(x):
    return np.sum((x - 5.0) ** 2)


class LinearPolicy(object):
    @staticmethod
    def from_flat_arrays(array, in_size, out_size):
        array = np.array(array)
        w, b = array[:, :-out_size], array[:, -out_size:]
        w = np.reshape(w, [-1, out_size, in_size])
        return LinearPolicy(w=w, b=b)

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def sample_action(self, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        # action = np.matmul(self.w, inputs) + self.b
        action = np.einsum("ijk,ik->ij", self.w, inputs) + self.b
        action = np.reshape(action, [-1, 3])
        return action


# TODO(mmatena): Make this cleaner.
IP_FILE = "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt"
with open(IP_FILE, "r") as f:
    ip = f.read()

# TODO(mmatena): Make this configurable.
conn = rpyc.connect(ip, 18861, config={"allow_all_attrs": True})
conn._config["sync_request_timeout"] = None
gym_service = conn.root


encoder = saved_models.raw_rollout_vae_32ld()
# sequence_model = saved_models.encoded_rollout_transformer()
sequence_model = transformer.base_deterministic_transformer()
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

#
# def batched_rollout(env, policy, max_steps, batch_size):
#     print("TODO: The handling for dones is incorrect!")
#     env.reset()
#     policy.initialize(env=env, max_steps=max_steps)

#     dones = batch_size * [False]

#     rollout = Rollout()
#     for step in range(max_steps):
#         # TODO(mmatena): Support environments without a "state_pixels" render mode.
#         start = time.time()
#         whether_to_renders = pickle.dumps([not d for d in dones])
#         obs = env.render(whether_to_renders)
#         # This might happen if we are running on a remote gym server using rpc.
#         if isinstance(obs, bytes):
#             obs = pickle.loads(obs)
#         print(f"Render time: {time.time() - start} s")

#         start = time.time()
#         action = policy.sample_action(obs=obs, step=step, rollout=rollout)
#         print(f"Sample action time: {time.time() - start} s")

#         start = time.time()
#         step_infos = env.step(pickle.dumps(action))
#         # This might happen if we are running on a remote gym server using rpc.
#         if isinstance(step_infos, bytes):
#             step_infos = pickle.loads(step_infos)
#         print(f"Env step time: {time.time() - start} s")

#         rollout.obs_l.append(obs)
#         rollout.action_l.append(action)
#         rollout.reward_l.append([si.reward for si in step_infos])

#         for i, si in enumerate(step_infos):
#             if si.done:
#                 dones[i] = True

#         if all(dones):
#             break

#     return [sum(s) for s in np.array(rollout.reward_l).T.tolist()]


def batched_rollout(env, policy, max_steps, batch_size):
    print("TODO: The handling for dones is incorrect!")
    env.reset()
    policy.initialize(env=env, max_steps=max_steps)

    dones = batch_size * [False]
    rollout = Rollout()
    for step in range(max_steps):
        if step == 0:
            # TODO(mmatena): Support environments without a "state_pixels" render mode.
            start = time.time()
            whether_to_renders = pickle.dumps([not d for d in dones])
            obs = env.render(whether_to_renders)
            # This might happen if we are running on a remote gym server using rpc.
            if isinstance(obs, bytes):
                obs = pickle.loads(zlib.decompress(obs))
            print(f"Render time: {time.time() - start} s")
            rollout.obs_l.append(obs)

        obs = rollout.obs_l[-1]

        start = time.time()
        action = policy.sample_action(obs=obs, step=step, rollout=rollout)
        print(f"Sample action time: {time.time() - start} s")

        start = time.time()
        step_infos = env.step(pickle.dumps(action))
        # This might happen if we are running on a remote gym server using rpc.
        if isinstance(step_infos, bytes):
            step_infos = pickle.loads(zlib.decompress(step_infos))
        print(f"Env step time: {time.time() - start} s")

        rollout.obs_l.append([si.observation for si in step_infos])
        rollout.action_l.append(action)
        rollout.reward_l.append([si.reward for si in step_infos])

        for i, si in enumerate(step_infos):
            if si.done:
                dones[i] = True

        if all(dones):
            break

    return [sum(s) for s in np.array(rollout.reward_l).T.tolist()]


def get_scores(solutions):
    linear_policy = LinearPolicy.from_flat_arrays(
        solutions, in_size=in_size, out_size=out_size
    )
    policy = policies.CarRacingPolicy(
        encoder=encoder,
        sequence_model=sequence_model,
        policy=linear_policy,
        max_seqlen=max_seqlen,
    )
    gym_service.make("CarRacing-v0", len(solutions))
    print("Increase MAX STEPS!!!!!")
    return batched_rollout(
        gym_service, policy, max_steps=100, batch_size=len(solutions)
    )


# es = cma.CMAEvolutionStrategy(8 * [0], 0.5, {"popsize": 64})
es = cma.CMAEvolutionStrategy(
    (in_size * out_size + out_size) * [0], 0.5, {"popsize": 64}
)

for i in range(2):
    start = time.time()
    solutions = es.ask()

    num_trials = 2
    args = functools.reduce(list.__add__, [num_trials * [s] for s in solutions])

    scores = get_scores(args)
    scores = misc.divide_chunks(scores, num_trials)
    fitlist = [sum(s) / num_trials for s in scores]

    es.tell(solutions, (np.array(fitlist)))

    print(f"CMA step time: {time.time() - start} s")


# CMA step time: 21 + 27 s
