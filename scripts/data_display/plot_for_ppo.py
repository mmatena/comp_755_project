R"""
Our 10k rollouts are comprised of a total of 5,212,576 timesteps.

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

flags.DEFINE_string("ppo_csv", "/home/owner/Downloads/progress-caveflyer.csv", "")
flags.DEFINE_string("controllers_csv", "/home/owner/Downloads/controllers.csv", "")
flags.DEFINE_integer("base_steps", 0, "")
# flags.DEFINE_integer("base_steps", 5_212_576, "")
flags.DEFINE_list(
    "controllers",
    [
        "deterministic_transformer_64dm_32di",
        "deterministic_transformer_64dm_32di_wide",
        "deterministic_transformer_256dm_32di",
        "no_history_64dk_ret4_half_stride",
    ],
    "",
)
flags.DEFINE_float("steps_per_cma_step", 64 * 16 * 5_212_576 / 10_000, "")

# flags.mark_flag_as_required("file")

COMMENT_MARKER = "#"


def smooth_series(series, factor=0.99, window=50):
    smoothed = np.empty([len(series)])
    for i in range(len(series)):
        section = series[max(i - window + 1, 0) : i + 1]
        weights = factor ** np.arange(len(section))[::-1]
        smoothed[i] = np.sum(section * weights) / np.sum(weights)
    return smoothed


def filter(x, y):
    mask = x < 5e7
    return x[mask], y[mask]


def get_controllers():
    controllers = {}
    with open(FLAGS.controllers_csv) as f:
        for row in csv.reader(f):
            if not len(row) or row[0].startswith(COMMENT_MARKER):
                continue
            name = row[0]
            data = np.array([float(x) for x in row[1:] if x])
            controllers[name] = data
    return controllers


def get_ppo_data():
    x, y = [], []
    with open(FLAGS.ppo_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(row["misc/total_timesteps"])
            y.append(row["eprewmean"])
    return np.array(x, dtype=np.float64), np.array(y, dtype=np.float64)


def plot_data(ppo, controllers):
    ppo_x, ppo_y = ppo

    plt.plot(ppo_x / 1e6, smooth_series(ppo_y), label="PPO")
    for controller in FLAGS.controllers:
        series = controllers[controller]
        series_x = FLAGS.base_steps + FLAGS.steps_per_cma_step * (
            1 + np.arange(len(series))
        )
        series_x, series = filter(series_x, series)
        plt.plot(series_x / 1e6, smooth_series(series), label=controller)

    plt.legend(loc="lower right")
    plt.xlabel("Time steps (M)")
    plt.ylabel("Average cumulative reward")
    plt.show()


def main(_):
    ppo = get_ppo_data()
    controllers = get_controllers()
    plot_data(ppo, controllers)


if __name__ == "__main__":
    app.run(main)
