"""Common code for generating the k-nn lookup data.
See https://arxiv.org/pdf/1911.00172.pdf for more details.
"""
import os
import time
import functools

from absl import app
from absl import flags
import tensorflow as tf

from rl755.common import misc
from rl755.common import structs
from rl755.data.car_racing import encoded_rollouts
from rl755.data.car_racing import processing
from rl755.models.car_racing import saved_models
from rl755.models.car_racing import transformer

FLAGS = flags.FLAGS

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
flags.DEFINE_string("out_dir", None, "The directory to write the tfrecords to.")
flags.DEFINE_string("out_name", None, "Prefix to give the generated tfrecord files.")

flags.DEFINE_integer(
    "num_passes",
    1,
    "The number of passes to make through the dataset.",
    lower_bound=1,
)
flags.DEFINE_integer("batch_size", 128, "Batch size.", lower_bound=1)

flags.mark_flag_as_required("num_outer_shards")
flags.mark_flag_as_required("outer_shard_index")
flags.mark_flag_as_required("num_sub_shards")
flags.mark_flag_as_required("gpu_index")
flags.mark_flag_as_required("out_dir")
flags.mark_flag_as_required("out_name")

SEQUENCE_LENGTH = 32


def get_model():
    model = saved_models.encoded_rollout_transformer()
    model.compile()
    return model


def get_dataset_files():
    files = (
        tf.io.matching_files(encoded_rollouts.TFRECORDS_PATTERN.format(split="train"))
        .numpy()
        .tolist()
    )
    # Ensure a consistent order.
    files.sort()
    return files


def get_dataset():
    files = get_dataset_files()
    files = misc.evenly_partition(files, FLAGS.num_outer_shards)[
        FLAGS.outer_shard_index
    ]

    files = tf.data.Dataset.from_tensor_slices(files)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        encoded_rollouts.parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.repeat(FLAGS.num_passes)
    ds = ds.map(
        functools.partial(processing.slice_example, slice_size=SEQUENCE_LENGTH + 1),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(
        functools.partial(
            transformer.to_ar_inputs_and_targets, sequence_length=SEQUENCE_LENGTH
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.batch(FLAGS.batch_size)
    return ds


def get_keys(model):
    # return model.transformer.encoder_layers[-1].self_attention_layer.output
    x = model.transformer
    x = x.encoder_layers[-1]
    x = x.self_attention_layer
    x = x.output
    return x


def get_keys_and_values(inputs, targets, key_model):
    keys = key_model(inputs, training=False)[:, -1]
    values = targets[:, -1]
    return keys, values


def run_shard(key_model, ds, sub_shard_index):

    ds = ds.prefetch(4)
    ds = ds.map(
        functools.partial(get_keys_and_values, key_model=key_model),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    num_total_shards = FLAGS.num_outer_shards * FLAGS.num_sub_shards
    total_shard_index = FLAGS.num_sub_shards * FLAGS.outer_shard_index + sub_shard_index

    filepath = os.path.join(
        FLAGS.out_dir,
        misc.sharded_filename(
            f"{FLAGS.out_name}.tfrecord",
            shard_index=total_shard_index,
            num_shards=num_total_shards,
        ),
    )
    with tf.io.TFRecordWriter(filepath) as file_writer:
        for keys, values in ds:
            for key, value in zip(keys, values):
                file_writer.write(
                    structs.key_value_to_tfrecord(key, value).SerializeToString()
                )


def main(_):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[FLAGS.gpu_index], "GPU")
    tf.config.experimental.set_memory_growth(gpus[FLAGS.gpu_index], True)

    ds = get_dataset()

    model = get_model()
    input_ = tf.keras.Input(shape=[SEQUENCE_LENGTH, 32 + 4 + 1])
    model(input_)
    key_model = tf.keras.Model(inputs=input_, outputs=get_keys(model))

    start = time.time()

    for i in range(FLAGS.num_sub_shards):
        run_shard(
            key_model=key_model,
            ds=ds.shard(num_shards=FLAGS.num_sub_shards, index=i),
            sub_shard_index=i,
        )

    end = time.time()
    print("Took", end - start, "seconds to run.")


if __name__ == "__main__":
    app.run(main)
