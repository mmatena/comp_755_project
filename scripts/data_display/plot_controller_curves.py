R"""
# TODO: Get the order and colors good.

# Transformer vs LSTM vs nothing:
python3 scripts/data_display/plot_controller_curves.py \
    --file=$HOME/Downloads/controllers.csv \
    --controllers=no_mem,deterministic_lstm_64dm_32di,deterministic_lstm_256dm_32di,deterministic_transformer_32dm_32di,deterministic_transformer_64dm_32di,deterministic_transformer_256dm_32di \
    --labels='No memory','LSTM $d_m=64$','LSTM $d_m=256$','Transformer $d_m=32$','Transformer $d_m=64$','Transformer $d_m=256$' \
    --linestyles=solid,dashed,solid,dotted,dashed,solid \
    --colors=darkblue,violet,purple,lightgreen,limegreen,darkgreen
    # plum,violet,purple

# Transformer with different sizes: (the @ get converted to commas)
python3 scripts/data_display/plot_controller_curves.py \
    --file=$HOME/Downloads/controllers.csv \
    --controllers=deterministic_transformer_64dm_32di,deterministic_transformer_64dm_32di_long,deterministic_transformer_64dm_32di_short,deterministic_transformer_64dm_32di_skinny,deterministic_transformer_64dm_32di_wide \
    --labels='$L=6@ d_{ff}=256$','$L=12@ d_{ff}=256$','$L=3@ d_{ff}=256$','$L=6@ d_{ff}=128$','$L=6@ d_{ff}=512$'

# Explicit retrieval:
python3 scripts/data_display/plot_controller_curves.py \
    --file=$HOME/Downloads/controllers.csv \
    --controllers=\
'episodic_32dk_ret4_half_stride',\
'no_history_32dk_ret4_half_stride',\
deterministic_transformer_32dm_32di,\
'episodic_64dk_ret4_half_stride',\
'no_history_64dk_ret4_half_stride',\
deterministic_transformer_64dm_32di \
    --labels=\
'Episodic retrieval@ $d_m = 32$',\
'No retrieval with extra training@ $d_m = 32$',\
'Baseline@ $d_m=32$',\
'Episodic retrieval@ $d_m = 64$',\
'No retrieval with extra training@ $d_m = 64$',\
'Baseline@ $d_m=64$' \
    --linestyles=solid,dashed,dotted,solid,dashed,dotted \
    --colors=purple,violet,plum,darkgreen,limegreen,lightgreen


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

import seaborn as sns
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_string("file", None, "")
flags.DEFINE_list("controllers", None, "")
flags.DEFINE_list("labels", None, "")
flags.DEFINE_list("colors", [], "")
flags.DEFINE_list("linestyles", [], "")

flags.DEFINE_boolean("smooth", True, "")

flags.mark_flag_as_required("file")
flags.mark_flag_as_required("controllers")
flags.mark_flag_as_required("labels")

COMMENT_MARKER = "#"


def smooth_series(series, factor=0.99, window=50):
    smoothed = np.empty([len(series)])
    for i in range(len(series)):
        section = series[max(i - window + 1, 0) : i + 1]
        weights = factor ** np.arange(len(section))[::-1]
        smoothed[i] = np.sum(section * weights) / np.sum(weights)
    return smoothed


def smooth_controllers(controllers, factor=0.99, window=50):
    return {
        k: smooth_series(v, factor=factor, window=window)
        for k, v in controllers.items()
    }


def get_controllers():
    controllers = {}
    with open(FLAGS.file) as f:
        for row in csv.reader(f):
            if not len(row) or row[0].startswith(COMMENT_MARKER):
                continue
            name = row[0]
            data = np.array([float(x) for x in row[1:] if x])
            controllers[name] = data
    return controllers


def plot_controllers(controllers):
    for i, (name, label) in enumerate(zip(FLAGS.controllers, FLAGS.labels)):
        series = controllers[name]
        color = FLAGS.colors[i] if FLAGS.colors else None
        linestyle = FLAGS.linestyles[i] if FLAGS.linestyles else None
        plt.plot(
            np.arange(len(series)),
            series,
            label=label,
            color=color,
            linestyle=linestyle,
        )
    plt.legend(loc="lower right")
    plt.xlabel("Generation")
    plt.ylabel("Average cumulative reward")
    plt.show()


def main(_):
    global raw_controllers, controllers

    FLAGS.labels = [s.replace("@", ",") for s in FLAGS.labels]
    assert len(FLAGS.controllers) == len(FLAGS.labels)

    raw_controllers = get_controllers()

    if FLAGS.smooth:
        controllers = smooth_controllers(raw_controllers)
    else:
        controllers = raw_controllers

    plot_controllers(controllers)


if __name__ == "__main__":
    app.run(main)
