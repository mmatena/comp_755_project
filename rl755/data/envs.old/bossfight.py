
"""Datasets for the bossfight environment."""
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder

import getpass

ENVIRONMENT = "bossfight"


class RawRollouts(RawImageRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF RAW ROLLOUTS!!!")
        if getpass.getuser() == "tgreer":
            return "/playpen-raid1/tgreer/rl/tmp/raw_bossfight/{split}/data.tfrecord*"
        raise("Please pick a directory for bossfight on pine, then add it to this file")


class TestEncodedRollouts(EncodedRolloutDatasetBuilder):
    def _environment(self):
        return ENVIRONMENT

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF ENCODED ROLLOUTS!!!")

        if getpass.getuser() == "tgreer":
            return "/playpen-raid1/tgreer/rl/tmp/enc_bossfight/{split}/data.tfrecord*"
        raise("Please pick a directory for bossfight on pine, then add it to this file")

    def representation_size(self):
        return 32
