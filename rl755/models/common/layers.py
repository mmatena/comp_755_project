"""Common layers to use."""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MixtureOfGaussiansLayer(tf.keras.layers.Layer):
    def __init__(self, dimensionality, num_components, name="mog", **kwargs):
        super().__init__(name=name, **kwargs)
        self.dimensionality = dimensionality
        self.num_components = num_components
        self.logits = self.add_weight(
            shape=[num_components], initializer="random_normal", trainable=True
        )
        self.cat_dist = tfd.Categorical(logits=self.logits)

    def _get_gauss_params(self, inputs):
        locs, scales = tf.split(inputs, num_or_size_splits=2, axis=-1)
        locs = tf.split(
            locs,
            num_or_size_splits=self.num_components * [self.dimensionality],
            axis=-1,
        )
        scales = tf.split(
            scales,
            num_or_size_splits=self.num_components * [self.dimensionality],
            axis=-1,
        )
        return locs, tf.nn.softplus(scales)

    def _get_gauss_components(self, inputs):
        locs, scales = self._get_gauss_params(inputs)
        return [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(locs, scales)
        ]

    def _get_mix_of_gauss_distribution(self, inputs):
        return tfd.Mixture(
            cat=self.cat_dist, components=self._get_gauss_components(inputs)
        )

    def call(self, inputs):
        return self._get_mix_of_gauss_distribution(inputs)
