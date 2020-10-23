"""Models that explicitly retrieve experiences."""
import tensorflow as tf

from .interface import MemoryComponent


class EpisodicRetriever(MemoryComponent):
    """Not as good as a golden retriever. Woof!"""

    def __init__(self, prediction_network, lookup_network, key_size, **kwargs):
        # TODO: Add docs, prediction_network and lookup_network are
        # MemoryComponents/ArTransformers themselves.
        super().__init__(**kwargs)
        self.prediction_network = prediction_network
        self.lookup_network = lookup_network
        self.key_size = key_size

        self.query_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="query_proj"
        )
        self.key_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="key_proj"
        )
