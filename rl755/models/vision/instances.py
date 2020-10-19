"""Specific instantiations of vision models."""
from . import vae


def vae_32d():
    # TODO: Add docs.
    model = vae.Vae(representation_size=32)
    return model
