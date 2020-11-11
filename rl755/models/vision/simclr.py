"""A variational autoencoder."""
import tensorflow as tf
import tensorflow_probability as tfp

from .interface import VisionComponent

tfd = tfp.distributions


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


class ClrLoss(tf.keras.losses.Loss):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, y_true, y_pred):
        LARGE_NUM = 1e9
        temperature = self.model.temperature
        hidden = tf.math.l2_normalize(y_pred, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

        loss_a = tf.compat.v1.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1)
        )
        loss_b = tf.compat.v1.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1)
        )
        loss = loss_a + loss_b
        return loss


class Clr(VisionComponent):
    """A Contrastive learning representation."""

    def __init__(self, representation_size, temperature=1.0, name="clr", **kwargs):
        super().__init__(name=name, **kwargs)
        self.representation_size = representation_size
        self.temperature = temperature
        self.encoder = _get_encoder(representation_size)
        self.head = _get_head(representation_size)

    def call(self, x, training=None):
        # split the two images, go through encode and head separately
        image1, image2 = tf.split(x, num_or_size_splits=2, axis=-1)
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
