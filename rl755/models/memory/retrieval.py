"""Models that explicitly retrieve experiences."""
import tensorflow as tf

from .interface import MemoryComponentWithHistory

_NEG_VALUE = -1e9


class EpisodicRetriever(MemoryComponentWithHistory):
    """Not as good as a golden retriever. Woof!"""

    def __init__(
        self,
        prediction_network,
        key_network,
        query_network,
        key_size,
        history_stride,
        num_retrieved,
        **kwargs
    ):
        # TODO: Add docs, prediction_network and key_network are
        # MemoryComponents/ArTransformers themselves.
        super().__init__(**kwargs)
        self.prediction_network = prediction_network
        self.key_network = key_network
        self.query_network = query_network
        self.key_size = key_size
        self.history_stride = history_stride
        self.num_retrieved = num_retrieved

        self.query_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="query_proj"
        )
        self.key_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="key_proj"
        )

        # These represent retrieving no documents.
        self.empty_key = None
        self.empty_value = None

    def _compute_keys(self, values, training, rep_key, position=-1):
        batch_size = tf.shape(values)[0]
        values_per_history = tf.shape(values)[1]
        sequence_length = tf.shape(values)[2]

        flat_values = tf.reshape(values, [-1, sequence_length, tf.shape(values)[-1]])
        keys = self.key_network.get_hidden_representation(
            flat_values, training=training, position=position, key=rep_key
        )
        keys = self.key_proj(keys)
        keys = tf.reshape(keys, [batch_size, values_per_history, self.key_size])
        return keys

    def _get_key_values(
        self, history, history_length, sequence_length, training, position=-1
    ):
        # TODO: Incorporate the history_length in this step prevent unnecessary evaluation.
        max_history_length = tf.shape(history)[-2]
        values_per_history = (
            1 + (max_history_length - sequence_length) // self.history_stride
        )

        # TODO: Make this faster.
        values = [
            history[
                :, k * self.history_stride : k * self.history_stride + sequence_length
            ]
            for k in range(values_per_history)
        ]
        values = tf.stack(values, axis=0)
        values = tf.transpose(values, perm=[1, 0, 2, 3])

        keys = self._compute_keys(
            values, training=training, position=position, rep_key="no_grads"
        )

        print("TODO: Append the empty/key values")

        return keys, values

    def _get_queries(self, inputs, training, position=-1):
        queries = self.query_network.get_hidden_representation(
            inputs, training=training, position=position
        )
        queries = self.query_proj(queries)
        return queries

    def _retrieve_train(self, inputs, history, history_length, training):
        sequence_length = tf.shape(inputs)[-2]

        keys, values = self._get_key_values(
            history,
            history_length=history_length,
            sequence_length=sequence_length,
            training=training,
        )
        queries = self._get_queries(inputs, training)

        scores = tf.einsum("bvi,bi->bv", keys, queries)

        value_ends = sequence_length * (
            tf.range(tf.shape(scores)[-1], dtype=tf.int32) + 1
        )
        valid_mask = tf.cast(
            tf.less_equal(value_ends[None, :], history_length[:, None]),
            dtype=tf.float32,
        )
        scores = scores * valid_mask + _NEG_VALUE * (1.0 - valid_mask)

        retrieved_scores_no_grad, retrieved_indices = tf.math.top_k(
            scores, k=self.num_retrieved
        )
        retrieved_values = tf.gather(values, retrieved_indices, axis=1, batch_dims=1)
        retrieved_values = tf.stop_gradient(retrieved_values)

        retrieved_keys = self._compute_keys(
            retrieved_values, training=training, rep_key="with_grads"
        )

        retrieved_scores = tf.einsum("bvi,bi->bv", retrieved_keys, queries)
        ignore_mask = tf.cast(
            tf.equal(retrieved_scores_no_grad, _NEG_VALUE), dtype=tf.float32
        )
        retrieved_scores = (
            1.0 - ignore_mask
        ) * retrieved_scores + ignore_mask * _NEG_VALUE

        return retrieved_values, retrieved_scores

    def call_train(self, inputs, history, history_length, mask=None, training=None):
        assert mask is None, "Not handling masks when training the retrieval model."
        sequence_length = tf.shape(inputs)[-2]
        batch_size = tf.shape(inputs)[0]

        values, scores = self._retrieve_train(
            inputs, history, history_length, training=training
        )

        print(
            "TODO: processing on values/inputs such as pos embeddings and type embeddings"
        )

        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.broadcast_to(inputs, tf.shape(values))

        prediction_inputs = tf.concat([values, inputs], axis=-2)
        prediction_inputs = tf.reshape(
            prediction_inputs, [-1, 2 * sequence_length, tf.shape(inputs)[-1]]
        )
        predictions = self.prediction_network.get_hidden_representation(
            prediction_inputs, training=training, position=-1
        )
        predictions = tf.reshape(
            predictions, [batch_size, self.num_retrieved, tf.shape(predictions)[-1]]
        )

        weights = tf.nn.softmax(scores, axis=-1)

        weighted_predictions = tf.einsum("bvi,bv->bi", predictions, weights)

        return weighted_predictions


# from rl755.models.memory.retrieval import *
if True:
    from rl755.models.memory import instances

    @tf.function
    def fn():
        sequence_length = 32
        key_size = 16
        history_stride = sequence_length // 2
        num_retrieved = 4

        prediction_network = instances.deterministic_transformer_32dm_32di()
        query_network = instances.deterministic_transformer_32dm_32di()
        key_network = instances.deterministic_transformer_32dm_32di()

        prediction_network(tf.zeros([1, 64, 32 + 15]))
        query_network(tf.zeros([1, 32, 32 + 15]))
        key_network(tf.zeros([1, 32, 32 + 15]))
        # prediction_network.build([32, 32 + 15])
        # query_network.build([32, 32 + 15])
        # key_network.build([32, 32 + 15])

        model = EpisodicRetriever(
            prediction_network=prediction_network,
            key_network=key_network,
            query_network=query_network,
            key_size=key_size,
            history_stride=history_stride,
            num_retrieved=num_retrieved,
        )

        inputs = tf.random.normal([2, 32, 32 + 15])
        history = tf.random.normal([2, 512, 32 + 15])
        history_length = tf.constant([311, 432], dtype=tf.int32)

        full_input = {
            "inputs": inputs,
            "history": history,
            "history_length": history_length,
        }

        outputs = model(full_input, training=True)
        return outputs

    fn()

# if True:
#     from rl755.data.envs.caveflyer import EncodedRolloutsVae32d
#     from itertools import islice

#     dsb = EncodedRolloutsVae32d()
#     ds = dsb.get_autoregressive_slices_with_full_history(
#         sequence_length=32, history_size=1024
#     )
#     for x in islice(ds.as_numpy_iterator(), 10):
#         print(x)
