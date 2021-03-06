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
    """Memory component models with access to full history should extend this."""

    def call(self, x, mask=None, training=None):
        return self._call(
            x["inputs"],
            history=x["history"],
            history_length=x["history_length"],
            mask=mask,
            training=training,
        )

    def _call(self, inputs, history, history_length, mask=None, training=None):
        raise NotImplementedError()

    def get_value_stride(self):
        raise NotImplementedError()

    def get_key_size(self):
        return NotImplementedError()

    def key_from_value(self, value, training=None):
        raise NotImplementedError()

    def get_hidden_representation(
        self,
        inputs,
        history,
        keys,
        history_length,
        num_keys,
        mask=None,
        training=None,
        position=-1,
    ):
        # TODO: Add docs.
        raise NotImplementedError()
