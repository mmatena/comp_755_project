"""Some general classes/interfaces around encoders."""
import tensorflow as tf


class ObservationEncoder(tf.keras.Model):
    """Abstract class that models that encode observations should extend."""

    # def __init__(self, representation_size=None, **kwargs):
    #     """Initializer that should be implemented.

    #     Args:
    #         representation_size: the size of the encoded representation.
    #             Usually a positive integer indicating its dimensionality, but
    #             there may be cases where you want an e.g. 2D representation.
    #             It's up to the implementing class and its users to decide that.
    #     """
    #     super().__init__(**kwargs)

    # def train_step(self, data):
    #     """The model must implement a custom train step.

    #     This comes from training VAEs. If other representation learning techniques
    #     can use a pre-built keras loss, then we should find a way to let them use
    #     that. That'll make support for data parallelism much easier.
    #     """
    #     raise NotImplementedError()

    # def compute_full_representation(self, x):
    #     """Returns a representation given a batch of raw observations.

    #     Args:
    #         x: a tf.Tensor containing a batch of raw observations
    #     Returns:
    #         Either a single tf.Tensor or a 2-tuple of a tf.Tensor and dict
    #         mapping strings to tf.Tensors.

    #         The tf.Tensor will be the representation as a single tensor. For
    #         deterministic encoders, this will just be the representation. For
    #         a VAE, this will be the mean of the posterior distribution.

    #         The dict, if present, allows for returning other information relevant
    #         to the representation. For example, this lets you return the
    #         standard deviation of the posterior distribution for a VAE.
    #     """
    #     raise NotImplementedError()
