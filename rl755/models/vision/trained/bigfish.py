"""Trained vision components for the bigfish environment."""
from .. import instances


def vae_32d():
    # TODO: Add docs
    weights_path = (
        "/pine/scr/m/m/mmatena/comp_755_project/models/vision/vae_32d/model-100.hdf5"
    )
    model = instances.vae_32d()
    model.load_weights(weights_path)
    return model
