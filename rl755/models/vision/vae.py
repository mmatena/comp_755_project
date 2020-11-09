"""A variational autoencoder."""
import tensorflow as tf
import tensorflow_probability as tfp

from .interface import VisionComponent

tfd = tfp.distributions
class PerPixelShift(tf.keras.layers.Layer):
    def __init__(self): 
        super(PerPixelShift, self).__init__()
    def build(self, input_shape):
        self.shift = self.add_weight(
            shape=(input_shape[1], input_shape[2], input_shape[3]),
            initializer=tf.keras.initializers.Zeros(),
            name="perPixelBias",
            trainable=True
        )
    def call(self, inputs):
        output = tf.add(self.shift, inputs)
        return output
    


def _get_encoder(representation_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(2 * representation_size),
        ]
    )


def _get_decoder(representation_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(representation_size,)),
            # No activation
            tf.keras.layers.Dense(units=1024),
            tf.keras.layers.Reshape(target_shape=(1, 1, 1024)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=5, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=5, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=6, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=6, strides=2, activation="sigmoid"
            ),
        ]
    )


class VaeLoss(tf.keras.losses.Loss):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, y_true, y_pred):
        posterior = self.model.posterior
        prior = self.model.prior
        loss_recon = tf.reduce_sum(tf.square(y_true - y_pred))
        loss_recon /= tf.cast(tf.shape(y_pred)[0], tf.float32)
        loss_kl = tf.reduce_mean(tfd.kl_divergence(posterior, prior))
        return loss_recon + self.model.beta * loss_kl


class Vae(VisionComponent):
    """A variational auto-encoder."""

    def __init__(self, representation_size, beta=1.0, name="vae", **kwargs):
        """Create a Vae.

        Args:
            representation_size: a positive integer, the size of the latent dimension
            beta: a float, the weight to give to the KL loss
            name: string, used by tf.keras.Model superclass
        """
        super().__init__(name=name, **kwargs)
        self.representation_size = representation_size
        self.beta = beta
        self.encoder = _get_encoder(representation_size)
        self.decoder = _get_decoder(representation_size)
        self.prior = tfd.MultivariateNormalDiag(
            tf.zeros(representation_size), tf.ones(representation_size)
        )

    def encode(self, x, training=None):
        mean, prevar = tf.split(
            self.encoder(x, training=training), num_or_size_splits=2, axis=-1
        )
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.nn.softplus(prevar))

    @tf.function
    def compute_full_representation(self, x, training=None):
        posterior = self.encode(x, training=training)
        return posterior.mean(), {"obs_std_devs": posterior.stddev()}

    def decode(self, z, training=None):
        logits = self.decoder(z, training=training)
        return logits

    def sample_unconditionally(self, num_samples=1):
        z = self.prior.sample(num_samples)
        return self.decode(z)

    def call(self, x, training=None):
        self.posterior = self.encode(x, training=training)
        return self.decode(self.posterior.sample(), training=training)

    def get_loss_fn(self):
        return VaeLoss(self)

    def get_representation_size(self):
        return self.representation_size
