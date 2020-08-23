"""I accidently saved just the weights.
This is jist to change it to saved model.
"""
import tensorflow as tf


WEIGHTS_PATH = "/pine/scr/m/m/mmatena/test_vae_train/model.hdf5"
MODEL_PATH = "/pine/scr/m/m/mmatena/test_vae_train/model.full.hdf5"

from rl755.models.car_racing.vae import Vae

vae = Vae(latent_dim=32, beta=1.0, log_losses=True)
vae.load_weights(WEIGHTS_PATH)
vae.save(MODEL_PATH)
