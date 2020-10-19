"""Some general classes/interfaces around encoders."""
import tensorflow as tf


class VisionComponent(tf.keras.Model):
    """Abstract class that models that encode observations should extend."""

    def compute_full_representation(self, x):
        """Returns a representation given a batch of raw observations.

        Args:
            x: a tf.Tensor containing a batch of raw observations
        Returns:
            Either a single tf.Tensor or a 2-tuple of a tf.Tensor and dict
            mapping strings to tf.Tensors.

            The tf.Tensor will be the representation as a single tensor. For
            deterministic encoders, this will just be the representation. For
            a VAE, this will be the mean of the posterior distribution.

            The dict, if present, allows for returning other information relevant
            to the representation. For example, this lets you return the
            standard deviation of the posterior distribution for a VAE.
        """
        raise NotImplementedError()

    def get_loss_fn(self):
        """Returns the loss function that will be used to train the encoder.

        Returns:
            An acceptable keras loss function. This can be a function taking
            in (y_true, y_pred) as arguments and returning a scalar loss tensor.
            It can also be a instance of a subclass of tf.keras.losses.Loss.
        """
        raise NotImplementedError()

    def get_representation_size(self):
        """Returns the size of the representation.

        Returns:
            A positive integer.
        """
        raise NotImplementedError()
