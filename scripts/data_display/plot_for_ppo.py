R"""

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

# flags.mark_flag_as_required("file")

COMMENT_MARKER = "#"


def get_ppo_data():
    x, y = [], []
    with open(FLAGS.ppo_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(row["misc/total_timesteps"])
            y.append(row["eprewmean"])
    return np.array(x, dtype=np.float64) / 1e6, np.array(y, dtype=np.float64)


def plot_data(ppo):
    ppo_x, ppo_y = ppo

    plt.plot(ppo_x, ppo_y, label="PPO")
    plt.legend(loc="lower right")
    plt.xlabel("Time steps (M)")
    plt.ylabel("Average cumulative reward")
    plt.show()


def main(_):
    ppo = get_ppo_data()
    plot_data(ppo)


if __name__ == "__main__":
    app.run(main)
