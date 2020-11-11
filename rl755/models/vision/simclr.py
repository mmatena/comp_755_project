"""A variational autoencoder."""
import functools

import tensorflow as tf

from .interface import VisionComponent
from rl755.data.common import processing

_LARGE_NUM = 1e9


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
            tf.keras.layers.Dense(representation_size),
        ]
    )


def _get_head(representation_size):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=16),
        ]
    )


def _sim(z1, z2):
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)
    return tf.einsum("...i,...i->...", z1, z2)


class ClrLoss(tf.keras.losses.Loss):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, y_true, y_pred):
        del y_true
        tau = self.model.temperature
        # Each is shaped [batched, d_z]
        z1, z2 = tf.split(y_pred, num_or_size_splits=2, axis=-1)

        numerator = -_sim(z1, z2) / tau

        z = tf.concat([z1, z2], axis=0)
        # Shape = [2N, 2N]
        denom = _sim(z[:, None, :], z[None, :, :]) / tau
        denom += -_LARGE_NUM * tf.eye(tf.shape(z)[0])
        denom = tf.reduce_logsumexp(denom, axis=-1)
        d1, d2 = tf.split(denom, num_or_size_splits=2, axis=0)

        return 0.5 * tf.reduce_mean(2 * numerator + d1 + d2)


class Clr(VisionComponent):
    """A Contrastive learning representation."""

    def __init__(self, representation_size, temperature=1.0, name="clr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.representation_size = representation_size
        self.temperature = temperature
        self.encoder = _get_encoder(representation_size)
        self.head = _get_head(representation_size)

    def _augment(self, x):
        map_fn = functools.partial(processing.augment_for_train, height=64, width=64)
        image1 = tf.vectorized_map(map_fn, x)
        image2 = tf.vectorized_map(map_fn, x)
        return image1, image2

    def call(self, x, training=None):
        # split the two images, go through encode and head separately
        # image1, image2 = tf.split(x, num_or_size_splits=2, axis=-1)
        image1, image2 = self._augment(x)
        rep1 = self.encoder(image1, training=training)
        hidden1 = self.head(rep1, training=training)
        rep2 = self.encoder(image2, training=training)
        hidden2 = self.head(rep2, training=training)
        # put the two images together
        hidden = tf.concat([hidden1, hidden2], -1)
        return hidden

    @tf.function
    def compute_full_representation(self, x, training=None):
        rep = self.encoder(x, training=training)
        return rep

    def get_loss_fn(self):
        return ClrLoss(self)

    def get_representation_size(self):
        return self.representation_size
