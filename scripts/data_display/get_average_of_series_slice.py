R"""

python3 scripts/data_display/get_average_of_series_slice.py \
    --file=$HOME/Downloads/controllers.csv \
    --num=500 \
    --num_is_start

python3 scripts/data_display/get_average_of_series_slice.py \
    --file=$HOME/Downloads/memory.csv \
    --num=5 \
    --num_is_start=False
"""
import csv
import io
import os
import pathlib
import pickle
import re

from absl import app
from absl import flags

import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("file", None, "")
flags.DEFINE_integer("num", None, "")
flags.DEFINE_boolean("num_is_start", False, "")

flags.mark_flag_as_required("file")
flags.mark_flag_as_required("num")

COMMENT_MARKER = "#"


def print_average(series):
    print("Name\tSlice length\tAverage")
    for name, values in series.items():
        if FLAGS.num_is_start:
            section = values[FLAGS.num :]
        else:
            section = values[-FLAGS.num :]
        length = len(section)
        if length == 0:
            average = "N/A"
        else:
            average = np.mean(section)
        print(f"{name}\t{length}\t{average}")


def get_series():
    series = {}
    with open(FLAGS.file) as f:
        for row in csv.reader(f):
            if not len(row) or row[0].startswith(COMMENT_MARKER):
                continue
            name = row[0]
            data = np.array([float(x) for x in row[1:] if x])
            series[name] = data
    return series


def main(_):
    series = get_series()
    print_average(series)


if __name__ == "__main__":
    app.run(main)
