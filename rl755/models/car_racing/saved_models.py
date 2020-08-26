"""Utility/book keeping to access models we've trained.

The reason this class exists is because I'm not sure of the best
practices for doing this yet.

We save the weights of the models only during training and store the
mapping between parameter files and model architectures here.

Doing it this way let's us get the tf.kera.models.Models subclass
object at the cost of having to keep track of the mapping between
saved parameters and architecture manually.
"""
from rl755.models.car_racing.vae import Vae


def raw_rollout_vae_32ld():
    """VAE trained on images from raw rollouts for 100k steps with
    a batch size of 256 for 100k steps and a latent dim of 32 and
    beta of 1.0. Used L2 reconstruction loss between pixels.
    """
    weights_path = "/pine/scr/m/m/mmatena/test_vae_train/model.hdf5"
    vae = Vae(latent_dim=32, beta=1.0)
    vae.load_weights(weights_path)
    return vae
