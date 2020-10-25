"""Models that explicitly retrieve experiences."""
import tensorflow as tf

from .interface import MemoryComponentWithHistory


class EpisodicRetriever(MemoryComponentWithHistory):
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


if True:
    from rl755.data.envs.caveflyer import EncodedRolloutsVae32d
    from itertools import islice

    dsb = EncodedRolloutsVae32d()
    ds = dsb.get_autoregressive_slices_with_full_history(
        sequence_length=32, history_size=1024
    )
    for x in islice(ds.as_numpy_iterator(), 10):
        print(x)
