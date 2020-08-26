"""Main file for running parallel rollouts of CarRacing-v0."""
import functools
import os
import pickle
import tempfile
import time

from absl import app
from absl import flags
import ray
from pyvirtualdisplay import Display

from rl755.data_gen import gym_rollouts

FLAGS = flags.FLAGS

# TODO(mmatena): Probably add some way to specify a policy.
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

flags.mark_flag_as_required("num_rollouts")
flags.mark_flag_as_required("parallelism")
flags.mark_flag_as_required("max_steps")
flags.mark_flag_as_required("out_dir")


def pickle_rollout(rollout, out_dir):
    # TODO(mmatena): Find better way to write to disc. Perhaps write as a tf record?
    fd, path = tempfile.mkstemp(dir=out_dir, suffix=".pickle")
    with open(path, "wb") as f:
        pickle.dump([rollout], f)
    os.close(fd)


def main(_):
    # It looks like OpenAI gym requires some sort of display, so we
    # have to fake one.
    display = Display(visible=0, size=(400, 300))
    display.start()

    ray.init()

    policy = gym_rollouts.HastingsRandomPolicy(time_scale=200, magnitude_scale=1.7)

    start = time.time()

    gym_rollouts.parallel_rollouts(
        "CarRacing-v0",
        policy=policy,
        max_steps=FLAGS.max_steps,
        num_rollouts=FLAGS.num_rollouts,
        process_rollout_fn=functools.partial(pickle_rollout, out_dir=FLAGS.out_dir),
        parallelism=FLAGS.parallelism,
    )

    end = time.time()
    print("Took", end - start, "seconds to run.")


if __name__ == "__main__":
    app.run(main)
