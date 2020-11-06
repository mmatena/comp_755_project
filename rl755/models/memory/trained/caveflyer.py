"""Trained memory models for the caveflyer environment."""
import tensorflow as tf

from .. import instances
from .. import instances_with_history

ACTION_SIZE = 15

_WEIGHTS_PATTERN = (
    "/pine/scr/m/m/mmatena/comp_755_project/models/memory/caveflyer/{}/model-{}.hdf5"
)


def deterministic_transformer_32dm_32di(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_transformer_32dm_32di", "100")
    model = instances.deterministic_transformer_32dm_32di(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_transformer_64dm_32di", "100")
    model = instances.deterministic_transformer_64dm_32di(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_256dm_32di(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_256dm_32di", "100"
    )
    model = instances.deterministic_transformer_256dm_32di(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_long(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_long", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_long(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_short(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_short", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_short(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_skinny(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_skinny", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_skinny(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_wide(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_wide", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_wide(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_short_skinny(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_short_skinny", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_short_skinny(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_transformer_64dm_32di_long_wide(**kwargs):
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format(
        "deterministic_transformer_64dm_32di_long_wide", "100"
    )
    model = instances.deterministic_transformer_64dm_32di_long_wide(**kwargs)
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_32dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_32dm_32di", "150")
    model = instances.deterministic_lstm_32dm_32di()
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_64dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_64dm_32di", "150")
    model = instances.deterministic_lstm_64dm_32di()
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def deterministic_lstm_256dm_32di():
    # TODO: Add docs
    weights_path = _WEIGHTS_PATTERN.format("deterministic_lstm_256dm_32di", "150")
    model = instances.deterministic_lstm_256dm_32di()
    # Build the model.
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    model.load_weights(weights_path)
    return model


def no_mem():
    return instances.no_mem()


def deterministic_transformer_32dm_32di_untrained(**kwargs):
    print("NOTE: THIS IS UNTRAINED!!!")
    model = instances.deterministic_transformer_32dm_32di(**kwargs)
    model(tf.keras.Input([None, 32 + ACTION_SIZE]))
    return model


def episodic_32dk_ret4_half_stride():
    print("TODO: THIS IS NOT TRAINED!!!")
    model = instances_with_history.episodic_32dk_ret4_half_stride(
        deterministic_transformer_32dm_32di_untrained
    )
    return model
