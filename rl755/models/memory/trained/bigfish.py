"""Trained memory models for the bigfish environment."""
import tensorflow as tf

from .. import instances

ACTION_SIZE = 15


def deterministic_transformer_32dm_32di():
    # TODO: Add docs
    print("TODO: USE A CHECKPOINT THAT HAS BEEN TRAINED LONGER!!!")
    weights_path = "/pine/scr/m/m/mmatena/comp_755_project/models/memory/deterministic_transformer_32dm_32di/model-075.hdf5"
    model = instances.deterministic_transformer_32dm_32di()
    # Build the model.
    model(tf.zeros([384, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model
