"""Some general interfaces around memory components."""
import tensorflow as tf


class MemoryComponent(tf.keras.Model):
    """Abstract class that memory component models should extend."""

    def get_loss_fn(self):
        """Returns the loss function used when training autoregressively.

        Returns:
            A loss function, typically an instance of tf.keras.losses.<LossFn>.
        """
        raise NotImplementedError()

    def get_hidden_representation(self, x, mask=None, training=None, position=-1):
        # TODO: Add docs.
        raise NotImplementedError()

    def get_representation_size(self):
        """Returns the size of the representation.

        Returns:
            A positive integer.
        """
        raise NotImplementedError()


class MemoryComponentWithHistory(MemoryComponent):
    # TODO: Add docs, maybe some more methods.
    pass
