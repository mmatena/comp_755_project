"""Trained vision components for the caveflyer environment."""
from .. import instances


def vae_32d():
    # TODO: Add docs
    weights_path = "/pine/scr/m/m/mmatena/comp_755_project/models/vision/caveflyer/vae_32d/model-100.hdf5"
    model = instances.vae_32d()
    model.load_weights(weights_path)
    return model

def clr_32d():
    # TODO: Add docs
    weights_path = "/pine/scr/m/t/mtguo/comp_755_project/vision/clr_32d/model-060.hdf5"
    model = instances.clr_32d()
    model.load_weights(weights_path)
    return model

def vae_32d_untrained():
    # TODO: Add docs, This is mostly for testing stuff locally, where we do not care
    # about the actual performance but just want to make sure the code runs.
    model = instances.vae_32d()
    return model
