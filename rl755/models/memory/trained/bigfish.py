"""Trained memory models for the bigfish environment."""
import tensorflow as tf

from .. import instances

ACTION_SIZE = 15


def deterministic_transformer_32dm_32di():
    # TODO: Add docs
    weights_path = "/pine/scr/m/m/mmatena/comp_755_project/models/memory/bigfish/deterministic_transformer_32dm_32di/model-050.hdf5"
    model = instances.deterministic_transformer_32dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def no_mem():
    return instances.no_mem()
