"""Evaluate the best controller from a set of checkpoints."""
import csv
import io
import os
import pathlib
import pickle
from pydoc import locate
import re
import time

from absl import app
from absl import flags
from absl import logging

import numpy as np

from rl755.data_gen import gym_rollouts
from rl755.models.memory.interface import MemoryComponentWithHistory
from rl755.models.policy import PolicyWrapper, PolicyWrapperWithHistory

FLAGS = flags.FLAGS

flags.DEFINE_string("directory", None, "")

flags.DEFINE_string(
    "environment",
    None,
    "The environment we are training on.",
)

flags.DEFINE_string("controller", "LinearController", "")
flags.DEFINE_string("vision_model", None, "")
flags.DEFINE_string("memory_model", None, "")

flags.DEFINE_integer("sequence_length", 32, "", lower_bound=1)

flags.DEFINE_string("checkpoint_regex", r"^checkpoint-\d+\.pickle$", "")
flags.DEFINE_integer("eval_trials", 1024, "")
flags.DEFINE_integer("rollout_max_steps", 1000, "")
flags.DEFINE_integer("eval_every_n", 25, "")
flags.DEFINE_integer("max_simul_envs", -1, "Negative integers mean no limits.")

flags.mark_flag_as_required("directory")
flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("vision_model")
flags.mark_flag_as_required("memory_model")

ACTION_SIZE = 15
OUTFILE_NAME = "evaluation.csv"


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


def get_checkpoints_to_eval(directory):
    checkpoints = {}
    for file in os.listdir(directory):
        if not re.match(FLAGS.checkpoint_regex, file):
            continue
        with open(os.path.join(directory, file), "rb") as f:
            checkpoint = pickle.load(f)
        step = checkpoint["step"]
        if not (step % FLAGS.eval_every_n):
            continue
        fitlist = checkpoint["fitlist"]
        solutions = checkpoint["solutions"]
        best_solution = solutions[np.argmax(fitlist)]
        checkpoints[step] = best_solution
    return checkpoints


def eval_solution(solution, step, vision_model, memory_model):
    if isinstance(memory_model, MemoryComponentWithHistory):
        Policy = PolicyWrapperWithHistory
    else:
        Policy = PolicyWrapper

    Controller = get_controller_class()
    in_size, out_size = get_in_out_sizes(vision_model, memory_model)

    learned_policy = Controller.from_flat_arrays(
        solution[None], in_size=in_size, out_size=out_size
    )

    total_envs = FLAGS.eval_trials
    max_simul_envs = FLAGS.max_simul_envs
    if max_simul_envs < 0:
        max_simul_envs = total_envs

    rewards = []
    for i in range(total_envs // max_simul_envs):
        logging.info(
            f"Running trials {i*max_simul_envs + 1}-{(i+1)*max_simul_envs} for step {step}."
        )
        start = time.time()
        policy = Policy(
            vision_model=vision_model,
            memory_model=memory_model,
            learned_policy=learned_policy,
            max_seqlen=FLAGS.sequence_length,
        )
        # TODO: Maybe add some form of stopping if all are complete. IDK if this happens
        # often enough to be a benefit.
        rollout_state = gym_rollouts.perform_rollouts(
            env_name=FLAGS.environment,
            num_envs=max_simul_envs,
            policy=policy,
            max_steps=FLAGS.rollout_max_steps,
        )
        rewards.extend(rollout_state.get_first_rollout_total_reward().tolist())
        logging.info(f"Took {time.time() - start} s.")

    return np.mean(rewards)


def main(_):
    assert not (FLAGS.eval_trials % FLAGS.max_simul_envs)
    logging.info("Reading checkpoints from directory.")
    solution_per_step = get_checkpoints_to_eval(FLAGS.directory)
    logging.info("Checkpoints have been read.")

    logging.info("Loading vision model.")
    vision_model = get_vision_model()
    logging.info("Vision model has been read.")

    logging.info("Loading memory model.")
    memory_model = get_memory_model()
    logging.info("Vision model has been read.")

    outfile = os.path.join(FLAGS.directory, OUTFILE_NAME)
    with open(outfile, "a+") as f:
        f.write("Step,Reward\n")
        for step in sorted(solution_per_step.keys()):
            solution = solution_per_step[step]
            average_reward = eval_solution(solution, step, vision_model, memory_model)
            logging.log(f"Average cumulative reward for step {step}: {average_reward}")
            f.write(f"{step},{average_reward}\n")


if __name__ == "__main__":
    app.run(main)
