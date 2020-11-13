""" Because the VAE encodes the information most common in the training images, the information in the residue is both rare, 
indicating importance to gameplay, and sparse, which allows threshhold then average-pool to capture it efficiently.
This even allows information never seen by the vision model to be exploited by the controller."""
from .interface import VisionComponent 
from . import vision_trained
from . import instances 

import tensorflow as tf
import tensorflow_addons as tfa


class ResidualVaeLoss(tf.keras.losses.Loss):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.internal_vae_loss = model.residue_vae.get_loss_fn()

    def call(self, y_true, y_pred):
        return self.internal_vae_loss(y_true, y_pred)



class ResidualVae(VisionComponent):
    def __init__(self, environment, **kwargs):
        super().__init__(name="resvae", **kwargs)
        self.vae = vision_trained.vae_32d(environment, name="primaryvae")
        
        self.residue_vae = instances.vae_32d(environment, name="residuevae")
        self.residue_vae.beta=.01

    def compute_full_representation(self, x):
        ### x.shape is (N, 64, 64, 3)
        vae_representation = self.vae.encode(x).mean()
        image_compressed = self.vae.decode(vae_representation)

        residue = x - image_compressed
        residue = tf.stop_gradient(residue)
        self.residue = residue
        r_vae_representation = self.residue_vae.encode(residue).mean()

        return tf.concat([vae_representation, r_vae_representation], 1)

    def decode(self, z):
        return self.vae.decode(z[:32]) + self.residue_vae.decode(z[32:])

    def call(self, x, training):
        vae_representation = self.vae.encode(x).mean()
        image_compressed = self.vae.decode(vae_representation)

        residue = x - image_compressed
        residue = tf.stop_gradient(residue)
        self.residue = residue
        return self.residue_vae.call(residue, training=training) + tf.stop_gradient(image_compressed)

    def get_loss_fn(self):
        return ResidualVaeLoss(self)

    def get_representation_size(self):
        return 64

