"""Datasets that come from rollouts where some encoding has occured.

Most of the encoding was done by the VAE at
`rl755.models.car_racing.saved_models.raw_rollout_vae_32ld`.
"""
import functools
import config

import tensorflow as tf

from rl755.environments import Environments
from rl755.data.common import processing
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder


class VaeEncodedRollouts(EncodedRolloutDatasetBuilder):
    def _environment(self):
        return Environments.CAR_RACING

    def _tfrecords_pattern(self):
        # mmatena: I forgot the tfrecord suffix when writing the files.
        return (
            config.dataset_dir
            + "/car_racing/"
            + "encoded_rollouts/{split}/encoded_rollouts*"
        )

    def _additional_features(self):
        return {"observation_std_devs": tf.io.VarLenFeature(tf.float32)}

    def representation_size(self):
        return 32

    def _sample_observations(self, example):
        o = example["observations"]
        noise = example["observation_std_devs"] * tf.random.normal(shape=tf.shape(o))
        return o + noise
