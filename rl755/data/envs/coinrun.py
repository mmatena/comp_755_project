"""Datasets for the coinrun environment."""
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder

ENVIRONMENT = "coinrun"


class RawRollouts(RawImageRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF RAW ROLLOUTS!!!")

        return "/pine/scr/m/m/mmatena/tmp/raw_coinrun/{split}/data.tfrecord*"


class TestEncodedRollouts(EncodedRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF ENCODED ROLLOUTS!!!")

        return "/pine/scr/m/m/mmatena/tmp/enc_coinrun/{split}/data.tfrecord*"

    def representation_size(self):
        return 32
