"""Train a VAE on random images collected from the rollouts."""
import functools
from pydoc import locate
import time
from rl755.models.vision import vision_trained

from absl import app
from absl import flags
import tensorflow as tf

from rl755.common import misc
from rl755.common import tfrecords

OBSERVATION_SHAPE = (64, 64, 3)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "environment", None, "Which environment we are encoding observations for."
)
flags.DEFINE_string("vision_model", None, "")

flags.DEFINE_string("split", None, "The dataset split to use.")

flags.DEFINE_integer(
    "estimated_num_rollouts",
    None,
    "Estimate on the number of rollouts. Used only for naming shards.",
)
flags.DEFINE_integer("desired_shard_mb", 100, "")


flags.DEFINE_string("out_dir", None, "The directory to write the tfrecords to.")
flags.DEFINE_string("out_name", "data", "Prefix to give the generated tfrecord files.")

flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("vision_model")
flags.mark_flag_as_required("estimated_num_rollouts")
flags.mark_flag_as_required("split")
flags.mark_flag_as_required("out_dir")


def get_raw_rollouts_builder():
    dsb_cls = locate(f"rl755.data.generic.RawRollouts")
    return dsb_cls(FLAGS.environment)


def get_dataset_files():
    raw_rollouts = get_raw_rollouts_builder()
    files = raw_rollouts.get_tfrecord_files(split=FLAGS.split).numpy().tolist()
    # Ensure a consistent order so that each file is processed exactly once.
    files.sort()
    return files


def get_dataset():
    raw_rollouts = get_raw_rollouts_builder()
    files = get_dataset_files()
    files = tf.data.Dataset.from_tensor_slices(files)

    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=1,
    )
    ds = ds.map(
        functools.partial(raw_rollouts.parse_tfrecord, process_observations=True),
        num_parallel_calls=1,
    )

    return ds


def get_vision_model():
    model = getattr(vision_trained, FLAGS.vision_model)
    return model(FLAGS.environment)


@tf.function
def encode_rollout(model, rollout):
    raw_observations = tf.reshape(rollout["observations"], (-1,) + OBSERVATION_SHAPE)
    rep = model.compute_full_representation(raw_observations)
    if isinstance(rep, tuple):
        rep, extra = rep
    else:
        extra = {}
    ret = {
        "observations": rep,
        "actions": rollout["actions"],
        "rewards": rollout["rewards"],
        "done_step": rollout["done_step"],
    }
    ret.update(extra)
    return ret


def serialize_encoded_rollout(rollout):
    feature_list = {
        "observations": tfrecords.to_float_feature_list(rollout["observations"]),
        "rewards": tfrecords.to_float_feature_list(rollout["rewards"][:, None]),
        "actions": tfrecords.to_int64_feature_list(rollout["actions"][:, None]),
        "done_step": tfrecords.to_int64_feature_list([[rollout["done_step"]]]),
    }

    extra_features = {
        k: tfrecords.to_float_feature_list(v)
        for k, v in rollout.items()
        if k not in feature_list
    }
    feature_list.update(extra_features)

    example_proto = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(feature_list=feature_list)
    )
    return example_proto.SerializeToString()


def main(_):
    model = get_vision_model()
    ds = get_dataset()
    start = time.time()

    with tfrecords.FixedSizeShardedWriter(
        directory=FLAGS.out_dir,
        filename=f"{FLAGS.out_name}.tfrecord",
        total_count=FLAGS.estimated_num_rollouts,
        desired_shard_mb=FLAGS.desired_shard_mb,
    ) as record_writer:
        for rollout in ds:
            encoded = encode_rollout(model, rollout)
            encoded = {k: v.numpy() for k, v in encoded.items()}
            serialized_record = serialize_encoded_rollout(encoded)
            record_writer.write(serialized_record)

    end = time.time()
    print("Took", end - start, "seconds to run.")


if __name__ == "__main__":
    app.run(main)
