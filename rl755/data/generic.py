
"""Datasets for the bossfight environment."""
from rl755.data.common.rollout_datasets import EncodedRolloutDatasetBuilder
from rl755.data.common.rollout_datasets import RawImageRolloutDatasetBuilder

import getpass



class RawRollouts(RawImageRolloutDatasetBuilder):
    def __init__(self, environment):
        super(RawImageRolloutDatasetBuilder, self).__init__()
        self.environment = environment
 
    def _environment(self):
        return self.environment

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF RAW ROLLOUTS!!!")
        if getpass.getuser() == "tgreer":
            return "../tmp/" + self.environment + "/raw_rollouts/{split}/data.tfrecord*"
        raise("Please pick a directory for bossfight on pine, then add it to this file")


class EncodedRollouts(EncodedRolloutDatasetBuilder):
    def __init__(self, environment):
        super(EncodedRolloutDatasetBuilder, self).__init__()
        self.environment = environment
    def _environment(self):
        return self.environment

    def _tfrecords_pattern(self):
        print("TODO: THIS IS JUST A TEST VERSION OF ENCODED ROLLOUTS!!!")

        if getpass.getuser() == "tgreer":
            return "../tmp/" + self.environment + "/enc_rollouts/{split}/data.tfrecord*"
        raise("Please pick a directory for bossfight on pine, then add it to this file")

    def representation_size(self):
        return 32
