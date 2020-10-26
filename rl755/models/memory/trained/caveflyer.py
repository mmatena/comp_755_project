"""Trained memory models for the caveflyer environment."""
import tensorflow as tf

from .. import instances

ACTION_SIZE = 15

_WEIGHTS_PATTERN = (
    "/pine/scr/m/m/mmatena/comp_755_project/models/memory/caveflyer/{}/model-{}.hdf5"
)


def deterministic_transformer_32dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_transformer_32dm_32di", "100")
    model = instances.deterministic_transformer_32dm_32di()
    # Build the model.
    # model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_transformer_64dm_32di", "100")
    model = instances.deterministic_transformer_64dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_256dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_256dm_32di", "100"
    )
    model = instances.deterministic_transformer_256dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_32dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_32dm_32di", "100")
    model = instances.deterministic_lstm_32dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_64dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_64dm_32di", "150")
    model = instances.deterministic_lstm_64dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_256dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_256dm_32di", "150")
    model = instances.deterministic_lstm_256dm_32di()
    # Build the model.
    print("TODO: Figure out how to do this without knowing batch or sequence length.")
    model(tf.zeros([128, 32, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def no_mem():
    return instances.no_mem()
