import csv
import io
import os
import pathlib
import pickle
import re

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_list("directories", None, "")
flags.DEFINE_string("checkpoint_regex", r"^checkpoint-\d+\.pickle$", "")
flags.DEFINE_integer("max_steps", 1000, "")

flags.mark_flag_as_required("directories")


def extract_from_directory(directory):
    row = [pathlib.PurePath(directory).parent.name] + FLAGS.max_steps * [""]
    for file in os.listdir(directory):
        print(file)
        if not os.path.isfile(file):
            continue
        if not re.match(FLAGS.checkpoint_regex, file):
            continue
        with open(os.path.join(directory, file), "rb") as f:
            checkpoint = pickle.load(f)
        step = checkpoint["step"]
        max_score = max(checkpoint["fitlist"])
        row[step + 1] = max_score
    return row


def main(_):
    data = []
    for directory in FLAGS.directories:
        data.append(extract_from_directory(directory))

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(data)
    data_str = output.getvalue()
    print(data_str)


if __name__ == "__main__":
    app.run(main)
