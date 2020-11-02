
"""Trained memory models for the bossfight environment."""
import tensorflow as tf

from .. import instances

ACTION_SIZE = 15


def deterministic_transformer_32dm_32di():
    # TODO: Add docs
    weights_path = "../tmp/bossfight_memory_model/model-050.hdf5"
    model = instances.deterministic_transformer_32dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([1024, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model



def deterministic_lstm_64dm_32di():
    # TODO: Add docs
    weights_path = "../tmp/bossfight_memory_model/model-050.hdf5"
    model = instances.deterministic_lstm_64dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([1024, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model
