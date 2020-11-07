"""Trained memory models for the coinrun environment."""
from . import instances
import tensorflow as tf


ACTION_SIZE=15
def deterministic_transformer_32dm_32di(environment):
    # TODO: Add docs
    print("TODO: THIS MODEL ISN'T TRAINED. LOAD WEIGHTS!!!")
    model = instances.deterministic_transformer_32dm_32di()
    model(tf.keras.Input([32, 32 + ACTION_SIZE])) 
    weights_path = (
     f"../tmp/{environment}/memory/deterministic_transformer_32dm_32di/model-004.hdf5"
    )
    model.load_weights(weights_path)
    return model
