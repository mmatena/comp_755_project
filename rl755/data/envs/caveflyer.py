"""Datasets for the caveflyer environment."""
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder

ENVIRONMENT = "caveflyer"


class RawRollouts(RawImageRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        return "/pine/scr/m/m/mmatena/comp_755_project/data/caveflyer/raw_rollouts/{split}/data.tfrecord*"


class EncodedRolloutsVae32d(EncodedRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        return "/pine/scr/m/m/mmatena/comp_755_project/data/caveflyer/vae_32d_rollouts/{split}/data.tfrecord*"

    def representation_size(self):
        return 32
