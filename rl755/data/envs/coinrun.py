"""Datasets for the coinrun environment."""
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder

ENVIRONMENT = "coinrun"


class RawRollouts(RawImageRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        return "/pine/scr/m/m/mmatena/tmp/raw_coinrun/{split}/data.tfrecord*"
