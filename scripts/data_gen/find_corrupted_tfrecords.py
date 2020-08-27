import os

from absl import app
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("pattern", None, "Pattern of tf records to examine.")

flags.mark_flag_as_required("pattern")


def main(_):
    files = tf.io.matching_files(FLAGS.pattern).numpy().aslist()

    for file in files:
        try:
            ds = tf.data.TFRecordDataset(file)
            for _ in ds:
                pass
        except tf.errors.DataLossError:
            print("ERROR FOUND IN FILE", file)


if __name__ == "__main__":
    app.run(main)
