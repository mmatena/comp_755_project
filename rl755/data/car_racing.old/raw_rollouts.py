"""Datasets that come from raw rollouts."""
import functools
import tensorflow as tf
import config

from rl755.environments import Environments
from rl755.data.common import processing
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder


class RawRollouts(RawImageRolloutDatasetBuilder):
    def _environment(self):
        return Environments.CAR_RACING

    def _tfrecords_pattern(self):

        return (
            config.dataset_dir
            + "/car_racing/"
            + "raw_rollouts/{split}/raw_rollouts.tfrecord*"
        )
