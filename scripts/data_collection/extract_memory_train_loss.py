import csv
import io
import os
import pathlib
import pickle
import re

from absl import app
from absl import flags

from tensorflow.python.summary.summary_iterator import summary_iterator

FLAGS = flags.FLAGS

flags.DEFINE_list("directories", None, "")
flags.DEFINE_integer("max_epochs", 151, "")

flags.mark_flag_as_required("directories")

EVENTS_FILE_PATTERN = r"^events\.out\.tfevents\..+\.v2$"


def extract_from_directory(directory):
    train_dir = os.path.join(directory, "train")
    row = [pathlib.PurePath(directory).name] + FLAGS.max_epochs * [""]
    events_file = None
    for file in os.listdir(train_dir):
        if re.match(EVENTS_FILE_PATTERN, file):
            events_file = os.path.join(train_dir, file)
            break
    if not events_file:
        return row

    for item in summary_iterator(events_file):
        if not hasattr(item, "step") or not hasattr(item, "summary"):
            continue
        if not item.summary.value or item.summary.value[0].tag != "epoch_loss":
            continue
        row[item.step + 1] = item.summary.value[0].simple_value
    return row


def main(_):
    data = []
    for directory in FLAGS.directories:
        data.append(extract_from_directory(directory))

    output = io.StringIO()
    writer = csv.writer(output)
    for row in data:
        writer.writerow(row)
    data_str = output.getvalue()
    print(data_str)


if __name__ == "__main__":
    app.run(main)
