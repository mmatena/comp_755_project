"""Some general classes/interfaces around sequential models."""
import tensorflow as tf


class SequentialModel(tf.keras.Model):
    """Abstract class that sequential models should extend."""

    def get_loss_fn(self):
        """Returns the loss function used when training autoregressively.

        Returns:
            A loss function, typically an instance of tf.keras.losses.<LossFn>.
        """
        raise NotImplementedError()

    def get_hidden_representation(self, x, mask=None, training=None, position=-1):
        raise NotImplementedError()
