"""Main file for running parallel rollouts of CarRacing-v0."""
import functools
import os
import pickle
from pydoc import locate
import tempfile
import time

from absl import app
from absl import flags
import ray
from pyvirtualdisplay import Display

from rl755 import models
from rl755.data_gen import gym_rollouts
from rl755.environments import EnvironmentInfos


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "num_rollouts",
    None,
    "The gross total number of rollouts to generate.",
    lower_bound=1,
)
flags.DEFINE_integer(
    "parallelism", None, "The number of rollouts to do in parallel.", lower_bound=1
)
flags.DEFINE_integer(
    "max_steps", None, "The maximum number of steps in each rollout.", lower_bound=1
)
flags.DEFINE_string("out_dir", None, "The directory to write the pickled rollouts to.")
flags.DEFINE_enum(
    "environment",
    None,
    EnvironmentInfos.keys(),
    "Which environment (aka task) to do rollouts on.",
)
flags.DEFINE_string(
    "policy",
    None,
    "The name of the policy to use. Will be accessed via rl755.models.policies.$policy(env)."
    "It should subclass rl755.data_gen.gym_rollouts.Policy or be a function returning such an object."
    "It will be called with no arguments.",
)

flags.mark_flag_as_required("num_rollouts")
flags.mark_flag_as_required("parallelism")
flags.mark_flag_as_required("max_steps")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("policy")

def get_environment_info():
    return EnvironmentInfos[FLAGS.environment]

def pickle_rollout(rollout, out_dir):
    # TODO(mmatena): Find better way to write to disc. Perhaps write as a tf record?
    fd, path = tempfile.mkstemp(dir=out_dir, suffix=".pickle")
    with open(path, "wb") as f:
        pickle.dump([rollout], f)
    os.close(fd)

def get_policy(environment_info):
    policy = locate(f"rl755.models.policies.{FLAGS.policy}")
    if not(policy):
        raise("Policy Not Found")
    return policy()


def main(_):
    # It looks like OpenAI gym requires some sort of display, so we
    # have to fake one.
    display = Display(visible=0, size=(400, 300))
    display.start()

    ray.init()

    environment_info = get_environment_info()
    policy = get_policy(environment_info)

    qualified_out_dir = os.path.join(FLAGS.out_dir, environment_info.folder_name)
 
    if not os.path.exists(qualified_out_dir):
        os.makedirs(qualified_out_dir)
    start = time.time()

    gym_rollouts.parallel_rollouts(
        environment_info,
        policy=policy,
        max_steps=FLAGS.max_steps,
        num_rollouts=FLAGS.num_rollouts,
        process_rollout_fn=functools.partial(
            pickle_rollout, 
            out_dir=qualified_out_dir),
        parallelism=FLAGS.parallelism,
    )

    end = time.time()
    print("Took", end - start, "seconds to run.")


if __name__ == "__main__":
    app.run(main)
