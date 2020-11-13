"""Learns a simple policy using CMA."""
import collections
import functools
import os
import pickle
from pydoc import locate
import time
import zlib
import math

from absl import app
from absl import flags
from absl import logging
import cma
import numpy as np
import rpyc
import tensorflow as tf

from rl755.common import misc
from rl755.data_gen import gym_rollouts
from rl755.models.memory.interface import MemoryComponentWithHistory
from rl755.models.policy import PolicyWrapper, PolicyWrapperWithHistory

ACTION_SIZE = 15

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "environment",
    None,
    "The environment we are training on.",
)
flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)

flags.DEFINE_string("controller", "LinearController", "")

flags.DEFINE_string("vision_model", None, "")
flags.DEFINE_string("memory_model", None, "")

flags.DEFINE_integer("sequence_length", 32, "", lower_bound=1)


flags.DEFINE_integer("cma_population_size", 32, "", lower_bound=1)
flags.DEFINE_integer("cma_trials_per_member", 12, "", lower_bound=1)
flags.DEFINE_integer("cma_steps", 1000, "", lower_bound=1)

flags.DEFINE_integer("max_simul_envs", -1, "Negative integers mean no limits.")

flags.DEFINE_integer("rollout_max_steps", 1000, "", lower_bound=1)


flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("vision_model")
flags.mark_flag_as_required("memory_model")


def get_vision_model():
    model = locate(
        f"rl755.models.vision.trained.{FLAGS.environment}.{FLAGS.vision_model}"
    )
    return model()


def get_memory_model():
    model = locate(
        f"rl755.models.memory.trained.{FLAGS.environment}.{FLAGS.memory_model}"
    )
    return model()


def get_controller_class():
    return locate(f"rl755.models.controller.controllers.{FLAGS.controller}")


def get_in_out_sizes(vision_model, memory_model):
    in_size = (
        vision_model.get_representation_size() + memory_model.get_representation_size()
    )
    out_size = ACTION_SIZE
    return in_size, out_size


def get_scores(solutions, vision_model, memory_model, max_steps):
    if isinstance(memory_model, MemoryComponentWithHistory):
        Policy = PolicyWrapperWithHistory
    else:
        Policy = PolicyWrapper

    Controller = get_controller_class()
    in_size, out_size = get_in_out_sizes(vision_model, memory_model)

    learned_policy = Controller.from_flat_arrays(
        solutions, in_size=in_size, out_size=out_size
    )

    total_envs = len(solutions)
    max_simul_envs = FLAGS.max_simul_envs
    if max_simul_envs < 0:
        max_simul_envs = learned_policy.batch_size()

    rewards = []

    for i in range(int(math.ceil(total_envs / max_simul_envs))):
        sliced_policy = learned_policy.slice_batch(
            i * max_simul_envs, (i + 1) * max_simul_envs
        )

        policy = Policy(
            vision_model=vision_model,
            memory_model=memory_model,
            learned_policy=sliced_policy,
            max_seqlen=FLAGS.sequence_length,
        )
        # TODO: Maybe add some form of stopping if all are complete. IDK if this happens
        # often enough to be a benefit.
        rollout_state = gym_rollouts.perform_rollouts(
            env_name=FLAGS.environment,
            num_envs=sliced_policy.batch_size(),
            policy=policy,
            max_steps=FLAGS.rollout_max_steps,
        )
        rewards.extend(rollout_state.get_first_rollout_total_reward().tolist())

    return rewards


def save_checkpoint(step, solutions, fitlist):
    obj = {
        "step": step,
        "solutions": solutions,
        "fitlist": fitlist,
    }
    checkpoint_file = os.path.join(FLAGS.model_dir, f"checkpoint-{step:03d}.pickle")
    with open(checkpoint_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main(_):
    # Needed for some reason when running on my laptop. IDK if its needed for longleaf,
    # but I'm putting it here just to be safe.
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    num_trials = FLAGS.cma_trials_per_member
    rollout_max_steps = FLAGS.rollout_max_steps

    vision_model = get_vision_model()
    memory_model = get_memory_model()

    in_size, out_size = get_in_out_sizes(vision_model, memory_model)

    Controller = get_controller_class()
    controller_params_count = Controller.get_parameter_count(
        in_size=in_size, out_size=out_size
    )

    es = cma.CMAEvolutionStrategy(
        controller_params_count * [0], 0.5, {"popsize": FLAGS.cma_population_size}
    )
    for step in range(FLAGS.cma_steps):
        start = time.time()
        solutions = es.ask()
        args = functools.reduce(list.__add__, [num_trials * [s] for s in solutions])

        scores = get_scores(
            args,
            vision_model=vision_model,
            memory_model=memory_model,
            max_steps=rollout_max_steps,
        )
        scores = misc.divide_chunks(scores, num_trials)

        fitlist = np.array([sum(s) / num_trials for s in scores])

        save_checkpoint(step=step, solutions=solutions, fitlist=fitlist)

        # We take the negative since our CMA is trying to reduce a loss.
        es.tell(solutions, -fitlist)

        logging.info(f"CMA step {step}:")
        logging.info(f"    time: {time.time() - start} s")
        logging.info(f"    max score: {max(fitlist)}")


if __name__ == "__main__":
    app.run(main)
