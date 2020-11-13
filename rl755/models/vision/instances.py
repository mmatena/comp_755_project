"""Specific instantiations of vision models."""
from . import vae
from . import residualvae

def vae_32d(environment, **kwargs):
    # TODO: Add docs.
    model = vae.Vae(representation_size=32, **kwargs)
    return model
def vae_64d(environment, **kwargs):
    # TODO: Add docs.
    model = vae.Vae(representation_size=64, **kwargs)
    return model

def residual_vae_64d(environment, **kwargs):
    model = residualvae.ResidualVae(environment)
    return model
