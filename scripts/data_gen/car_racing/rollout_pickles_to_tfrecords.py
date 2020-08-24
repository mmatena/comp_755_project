"""Convert the pickles generated by parallel_rollout_main.py to tf records."""
import glob
import os
import pickle
import tempfile
import uuid as uuid_module

from absl import app
from absl import flags
import ray
import tensorflow as tf

from rl755.common import structs
from rl755.common.misc import divide_chunks, evenly_partition, sharded_filename

FLAGS = flags.FLAGS

flags.DEFINE_integer('parallelism', None, 'Number of processes to use for the conversion.',
                     lower_bound=1)
# TODO(mmatena): Compute this based on desired shard size. For 2000 cartpole rounds
#                it seems like 6 is a good number. We want shards to be around 100-200 MB.
flags.DEFINE_integer('pickles_per_tfrecord_file', None,
                     'Number of pickled files to put in each tfrecord file.',
                     lower_bound=1)
flags.DEFINE_string('pickle_dir', None, 'The directory containing the pickle files.')
flags.DEFINE_string('out_dir', None, 'The directory to write the tfrecords to.')
flags.DEFINE_string('out_name', None, 'Prefix to give the generated tfrecord files.')

flags.mark_flag_as_required('parallelism')
flags.mark_flag_as_required('pickles_per_tfrecord_file')
flags.mark_flag_as_required('pickle_dir')
flags.mark_flag_as_required('out_dir')
flags.mark_flag_as_required('out_name')


@ray.remote
def convert_to_tf_records(filepaths, out_dir, pickles_per_tfrecord_file, uuid):
  for files in divide_chunks(filepaths, pickles_per_tfrecord_file):
    fd, record_file = tempfile.mkstemp(dir=out_dir, suffix=f"-{uuid}.tfrecord")
    with tf.io.TFRecordWriter(record_file) as writer:
      for file in files:
        rollouts = pickle.load(open(file, "rb"))
        for rollout in rollouts:
          record = structs.raw_rollout_to_tfrecord(rollout)
          writer.write(record.SerializeToString())
    os.close(fd)


def read_pickle_filenames(pickle_dir):
  return glob.glob(os.path.join(pickle_dir, "*.pickle"))


def rename_tfrecords(out_dir, out_name, uuid):
  records = glob.glob(os.path.join(out_dir, f"*-{uuid}.tfrecord"))
  for i, file in enumerate(records):
    base_name = sharded_filename(f'{out_name}.tfrecord', shard_index=i, num_shards=len(records))
    new_name = os.path.join(out_dir, base_name)
    os.rename(file, new_name)


def main(_):
  ray.init()

  # Use the UUID to mark the tfrecords we write as part of this program,
  # so as not to mess with existing tfrecords in the directory.
  uuid = uuid_module.uuid4().hex

  pickle_filenames = read_pickle_filenames(FLAGS.pickle_dir)

  # TODO(mmatena): Make this operation aware of the pickles_per_tfrecord_file.
  partitioned_filenames = evenly_partition(pickle_filenames, FLAGS.parallelism)

  futures = [
      convert_to_tf_records.remote(paths,
                                   out_dir=FLAGS.out_dir,
                                   pickles_per_tfrecord_file=FLAGS.pickles_per_tfrecord_file,
                                   uuid=uuid)
      for paths in partitioned_filenames
  ]
  ray.get(futures)

  rename_tfrecords(FLAGS.out_dir, out_name=FLAGS.out_name, uuid=uuid)


if __name__ == '__main__':
  app.run(main)
