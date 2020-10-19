"""Learns a simple policy using CMA."""
import collections
import functools
import os
import pickle
from pydoc import locate
import time
import zlib

from absl import app
from absl import flags
import cma
import numpy as np
import rpyc
import tensorflow as tf

from rl755.common import misc
from rl755.data_gen import gym_rollouts
from rl755.models.common.learned_policy import PolicyWrapper
from rl755.common.structs import Rollout, StepInfo
from rl755.environments import Environments

_ENV_NAME_TO_ENV = Environments.__members__

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "environment",
    None,
    _ENV_NAME_TO_ENV.keys(),
    "The environment we are training on.",
)
flags.DEFINE_string(
    "model_dir", None, "The directory to write checkpoints and logs to."
)

flags.DEFINE_string("learned_policy", "common.learned_policy.LinearPolicy", "")
flags.DEFINE_integer(
    "learned_policy_in_size",
    None,
    "",
    lower_bound=1,
)

flags.DEFINE_string("vision_model", None, "")
flags.DEFINE_string("sequence_model", None, "")

flags.DEFINE_integer("sequence_length", 32, "", lower_bound=1)


flags.DEFINE_integer("cma_population_size", 8, "", lower_bound=1)
flags.DEFINE_integer("cma_trials_per_member", 6, "", lower_bound=1)
flags.DEFINE_integer("cma_steps", 250, "", lower_bound=1)

flags.DEFINE_integer("rollout_max_steps", 1000, "", lower_bound=1)

flags.DEFINE_string(
    "gym_server_ip_file",
    "/pine/scr/m/m/mmatena/tmp/gym_server_ip.txt",
    "",
)
flags.DEFINE_integer("gym_server_port", 18861, "")

flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model_dir")
flags.mark_flag_as_required("learned_policy_in_size")
flags.mark_flag_as_required("vision_model")
flags.mark_flag_as_required("sequence_model")


def get_gym_ip():
    with open(FLAGS.gym_server_ip_file, "r") as f:
        ip = f.read()
    return ip


def get_gym_service():
    conn = rpyc.connect(
        get_gym_ip(), FLAGS.gym_server_port, config={"allow_pickle": True}
    )
    conn._config["sync_request_timeout"] = None
    gym_service = conn.root
    return conn, gym_service


def get_environment():
    return _ENV_NAME_TO_ENV[FLAGS.environment]


def get_vision_model():
    environment = get_environment()
    model = locate(f"rl755.models.{environment.folder_name}.{FLAGS.vision_model}")
    return model()


def get_sequence_model():
    environment = get_environment()
    model = locate(f"rl755.models.{environment.folder_name}.{FLAGS.sequence_model}")
    return model()


def get_learned_policy_cls():
    return locate(f"rl755.models.{FLAGS.learned_policy}")


def batched_rollout(env, policy, max_steps, batch_size):
    env.reset()
    policy.initialize(env=env, max_steps=max_steps)

    done_steps = max_steps * np.ones([batch_size], dtype=np.int32)
    rollout = Rollout()
    for step in range(max_steps):
        if step == 0:
            obs = env.render()
            obs = pickle.loads(obs)
            rollout.obs_l.append(obs)

        obs = rollout.obs_l[-1]

        action = policy.sample_action(obs=obs, step=step, rollout=rollout)

        step_infos = env.step(pickle.dumps(action))
        step_infos = pickle.loads(step_infos)

        rollout.obs_l.append(step_infos.observation)
        rollout.action_l.append(action)
        rollout.reward_l.append(step_infos.reward)

        done_upper_bounds = (
            step_infos.done * (step + 1) + (1 - step_infos.done) * max_steps
        )
        done_steps = np.minimum(done_steps, done_upper_bounds)

    rewards = np.array(rollout.reward_l).T.tolist()
    done_steps = done_steps.tolist()

    return [sum(r[:step]) for r, step in zip(rewards, done_steps)]


def get_scores(solutions, gym_service, vision_model, sequence_model, max_steps):
    environment = get_environment()
    LearnedPolicy = get_learned_policy_cls()
    in_size = FLAGS.learned_policy_in_size
    out_size = environment.action_shape[-1]

    learned_policy = LearnedPolicy.from_flat_arrays(
        solutions, in_size=in_size, out_size=out_size
    )
    policy = PolicyWrapper(
        vision_model=vision_model,
        sequence_model=sequence_model,
        learned_policy=learned_policy,
        max_seqlen=FLAGS.sequence_length,
    )
    gym_service.make(environment.open_ai_name, len(solutions))
    return batched_rollout(
        gym_service, policy, max_steps=max_steps, batch_size=len(solutions)
    )


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
    environment = get_environment()
    in_size = FLAGS.learned_policy_in_size
    out_size = environment.action_shape[-1]

    LearnedPolicy = get_learned_policy_cls()
    params_to_learn = LearnedPolicy.get_parameter_count(
        in_size=in_size, out_size=out_size
    )

    num_trials = FLAGS.cma_trials_per_member
    rollout_max_steps = FLAGS.rollout_max_steps

    vision_model = get_vision_model()
    sequence_model = get_sequence_model()

    conn, gym_service = get_gym_service()

    es = cma.CMAEvolutionStrategy(
        params_to_learn * [0], 0.5, {"popsize": FLAGS.cma_population_size}
    )
    # TODO: Probably occasionally disconnect and reconnect to gym server as it crashes
    # for some reason.
    for step in range(FLAGS.cma_steps):
        start = time.time()
        solutions = es.ask()
        args = functools.reduce(list.__add__, [num_trials * [s] for s in solutions])

        scores = get_scores(
            args,
            gym_service=gym_service,
            vision_model=vision_model,
            sequence_model=sequence_model,
            max_steps=rollout_max_steps,
        )
        scores = misc.divide_chunks(scores, num_trials)
        fitlist = np.array([sum(s) / num_trials for s in scores])

        save_checkpoint(step=step, solutions=solutions, fitlist=fitlist)

        # We take the negative since our CMA is trying to reduce a loss.
        es.tell(solutions, -fitlist)

        print(f"CMA step {step}:")
        print(f"    time: {time.time() - start} s")
        print(f"    max score: {max(fitlist)}")


if __name__ == "__main__":
    app.run(main)
