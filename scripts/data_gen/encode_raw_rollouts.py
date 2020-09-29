"""Train a VAE on random images collected from the rollouts."""
import functools
import os
import time

from absl import app
from absl import flags
import tensorflow as tf

from rl755 import data as data_module
from rl755 import models as models_module
from rl755.common import misc
from rl755.common import structs
from rl755.environments import Environments

_ENV_NAME_TO_ENV = Environments.__members__

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "environment",
    None,
    _ENV_NAME_TO_ENV.keys(),
    "Which environment we are encoding observations for.",
)
flags.DEFINE_string("model", None, "Name of model in `saved_models` to use.")

flags.DEFINE_string("split", None, "The dataset split to use.")
flags.DEFINE_string("out_dir", None, "The directory to write the tfrecords to.")
flags.DEFINE_string("out_name", None, "Prefix to give the generated tfrecord files.")

flags.DEFINE_integer(
    "num_outer_shards",
    None,
    "The number of times this binary is being called independently.",
    lower_bound=1,
)
flags.DEFINE_integer(
    "outer_shard_index", None, "The index in the runs of this binary.", lower_bound=0
)
flags.DEFINE_integer(
    "num_sub_shards",
    None,
    "The number of shards to generate for this part of the dataset. "
    "A good goal is to strive for shards of size 100-200 MB.",
    lower_bound=1,
)
flags.DEFINE_integer(
    "gpu_index",
    None,
    "The index of the GPU to use on multi-gpu systems.",
    lower_bound=0,
)

flags.mark_flag_as_required("num_outer_shards")
flags.mark_flag_as_required("outer_shard_index")
flags.mark_flag_as_required("num_sub_shards")
flags.mark_flag_as_required("gpu_index")
flags.mark_flag_as_required("split")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("out_name")
flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model")


def get_environment():
    return _ENV_NAME_TO_ENV[FLAGS.environment]


def get_raw_rollouts_builder():
    env = get_environment()
    return getattr(data_module, env.folder_name).raw_rollouts.RawRollouts()


def get_dataset_files():
    raw_rollouts = get_raw_rollouts_builder()
    files = raw_rollouts.get_tfrecord_files(split=FLAGS.split).numpy().tolist()
    # Ensure a consistent order so that each file is processed exactly once.
    files.sort()
    return files


def get_dataset(outer_shard_index, num_outer_shards, num_sub_shards):
    raw_rollouts = get_raw_rollouts_builder()
    files = get_dataset_files()
    files = misc.evenly_partition(files, num_outer_shards)[outer_shard_index]

    files = tf.data.Dataset.from_tensor_slices(files)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        functools.partial(raw_rollouts.parse_tfrecord, process_observations=True),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds


def load_model(model_name, environment):
    saved_models = getattr(models_module, environment.folder_name).saved_models
    return getattr(saved_models, model_name)()


def encode_map_fn(x, model):
    env = get_environment()
    raw_observations = tf.reshape(x["observations"], (-1,) + env.observation_shape)
    rep = model.compute_full_representation(raw_observations)
    if isinstance(rep, tuple):
        rep, extra = rep
    else:
        extra = {}
    ret = {
        "observations": rep,
        "actions": tf.reshape(x["actions"], (-1,) + env.action_shape),
        "rewards": tf.reshape(x["rewards"], (-1, 1)),
    }
    ret.update(extra)
    return ret


def run_shard(
    model,
    ds,
    out_dir,
    out_name,
    outer_shard_index,
    num_outer_shards,
    sub_shard_index,
    num_sub_shards,
):
    ds = ds.prefetch(1)
    ds = ds.map(
        functools.partial(encode_map_fn, model=model),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    num_total_shards = num_outer_shards * num_sub_shards
    total_shard_index = num_sub_shards * outer_shard_index + sub_shard_index

    filepath = os.path.join(
        out_dir,
        misc.sharded_filename(
            f"{out_name}.tfrecord",
            shard_index=total_shard_index,
            num_shards=num_total_shards,
        ),
    )
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for x in ds:
            file_writer.write(
                structs.encoded_rollout_to_tfrecord(x).SerializeToString()
            )


def main(_):
    environment = get_environment()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu_index], "GPU")
    tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu_index], True)

    model = load_model(FLAGS.model, environment)

    ds = get_dataset(
        outer_shard_index=FLAGS.outer_shard_index,
        num_outer_shards=FLAGS.num_outer_shards,
        num_sub_shards=FLAGS.num_sub_shards,
    )

    start = time.time()

    for i in range(FLAGS.num_sub_shards):
        run_shard(
            model=model,
            ds=ds.shard(num_shards=FLAGS.num_sub_shards, index=i),
            out_dir=FLAGS.out_dir,
            out_name=FLAGS.out_name,
            outer_shard_index=FLAGS.outer_shard_index,
            num_outer_shards=FLAGS.num_outer_shards,
            sub_shard_index=i,
            num_sub_shards=FLAGS.num_sub_shards,
        )

    end = time.time()
    print("Took", end - start, "seconds to run.")


if __name__ == "__main__":
    app.run(main)
