"""Create and save a bunch of rollouts."""
from pydoc import locate
import time

from absl import app
from absl import flags
from absl import logging

from rl755.common import tfrecords
from rl755.data_gen import gym_rollouts

FLAGS = flags.FLAGS

# TODO(mmatena): Add docs.
flags.DEFINE_string("environment", None, "")
flags.DEFINE_integer("num_rollouts", None, "The total number of rollouts to generate.")

flags.DEFINE_string(
    "policy",
    "policy.UniformRandomPolicy",
    "",
)

flags.DEFINE_integer("max_steps", None, "The maximum number of steps in each rollout.")

flags.DEFINE_integer("max_simul_envs", 256, "")
flags.DEFINE_integer("desired_shard_mb", 100, "")

flags.DEFINE_string("out_dir", None, "The directory to write the tfrecords to.")
flags.DEFINE_string("out_name", "data", "Prefix to give the generated tfrecord files.")


flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("num_rollouts")
flags.mark_flag_as_required("max_steps")
flags.mark_flag_as_required("out_dir")


def get_policy():
    policy_cls = locate(f"rl755.models.{FLAGS.policy}")
    return policy_cls()


def main(_):
    max_simul_envs = FLAGS.max_simul_envs
    max_steps = FLAGS.max_steps
    env_name = FLAGS.environment

    policy = get_policy()

    with tfrecords.FixedSizeShardedWriter(
        directory=FLAGS.out_dir,
        filename=f"{FLAGS.out_name}.tfrecord",
        total_count=FLAGS.num_rollouts,
        desired_shard_mb=FLAGS.desired_shard_mb,
    ) as record_writer:

        rollouts_remaining = FLAGS.num_rollouts
        step = 0
        while rollouts_remaining > 0:
            step_start_time = time.time()
            num_envs = min(max_simul_envs, rollouts_remaining)

            rollout_state = gym_rollouts.perform_rollouts(
                env_name=env_name, num_envs=num_envs, policy=policy, max_steps=max_steps
            )

            serialized_records = rollout_state.to_serialized_tfrecords()
            record_writer.write(serialized_records)

            logging.info(f"Step {step} took {time.time() - step_start_time} s")

            rollouts_remaining -= max_simul_envs
            step += 1


if __name__ == "__main__":
    app.run(main)
