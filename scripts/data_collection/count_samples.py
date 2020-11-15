"""Prints the number of samples from the first episode in each rollout."""
from pydoc import locate

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("environment", None, "")

flags.mark_flag_as_required("environment")


def get_rollout_length_fn(x):
    return x["done_step"]


def get_dataset():
    dsb_cls = locate(f"rl755.data.envs.{FLAGS.environment}.RawRollouts")
    ds = dsb_cls().rollouts_ds(process_observations=False, repeat=False)
    ds = ds.map(get_rollout_length_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def main(_):
    ds = get_dataset()
    total = 0
    for i, rollout_length in enumerate(ds.as_numpy_iterator()):
        total += rollout_length
        if not (i % 250):
            print(f"Rollout {i}: running total {total}")
    print(f"Final total: {total}")


if __name__ == "__main__":
    app.run(main)
