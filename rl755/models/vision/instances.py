"""Specific instantiations of vision models."""
from . import vae, simclr


def vae_32d():
    # TODO: Add docs.
    model = vae.Vae(representation_size=32)
    return model

def clr_32d():
    model = simclr.Clr(representation_size=32, temperature=0.5)
    return model