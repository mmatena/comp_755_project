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

        self._original_prediction_pos_embeddings = prediction_network.pos_embeddings
        print("TODO: It looks like we don't get gradients from these.")
        prediction_network.pos_embeddings = self._create_prediction_pos_embeddings()

        self.query_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="query_proj"
        )
        self.key_proj = tf.keras.layers.Dense(
            units=key_size, activation=None, name="key_proj"
        )

    def build(self, input_shape):
        print("TODO: Add a build_train and a build_no_train to the interface.")
        # These represent retrieving no documents.
        self.empty_key = self.add_weight(
            shape=[self.key_size],
            initializer="random_normal",
            trainable=True,
            name="empty_key",
        )
        self.empty_value = self.add_weight(
            shape=input_shape["inputs"][1:],
            initializer="random_normal",
            trainable=True,
            name="empty_value",
        )
        # super().build(input_shape)

    def _create_prediction_pos_embeddings(self):
        # Actually also includes type embeddings.
        hidden_size = self.prediction_network.transformer_params.hidden_size
        self.value_type_embedding = self.add_weight(
            shape=[hidden_size],
            initializer="random_normal",
            trainable=True,
            name="value_type_embedding",
        )
        self.input_type_embedding = self.add_weight(
            shape=[hidden_size],
            initializer="zeros",
            trainable=True,
            name="input_type_embedding",
        )

        pos_embeddings = self._original_prediction_pos_embeddings
        embedding_1 = pos_embeddings + self.value_type_embedding
        embedding_2 = pos_embeddings + self.input_type_embedding
        return tf.concat([embedding_1, embedding_2], axis=-2)

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

    def _extract_valid_values_func(self, history, history_length, sequence_length):
        # TODO: Incorporate the history_length in this step prevent unnecessary evaluation.
        max_history_length = history.shape[-2]
        values_per_history = (
            1 + (max_history_length - sequence_length) // self.history_stride
        )
        values = [
            history[
                :, k * self.history_stride : k * self.history_stride + sequence_length
            ]
            for k in range(values_per_history)
        ]
        values = tf.stack(values, axis=0)
        values = tf.transpose(values, perm=[1, 0, 2, 3])

        return values

    def _get_key_values(
        self, history, history_length, sequence_length, training, position=-1
    ):
        values = tf.py_function(
            self._extract_valid_values_func,
            inp=[history, history_length, sequence_length],
            Tout=tf.float32,
        )

        keys = self._compute_keys(
            values, training=training, position=position, rep_key="no_grads"
        )

        print("TODO: These variables do not get gradients. Redo!")
        empty_value = tf.broadcast_to(self.empty_value[None, ...], tf.shape(values))
        values = tf.concat([empty_value, values], axis=1)

        empty_key = tf.broadcast_to(self.empty_key[None, ...], tf.shape(keys))
        keys = tf.concat([empty_key, keys], axis=1)

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
            tf.less_equal(value_ends[None, :], history_length),
            dtype=tf.float32,
        )
        scores = scores * valid_mask + _NEG_VALUE * (1.0 - valid_mask)

        retrieved_scores_no_grad, retrieved_indices = tf.math.top_k(
            scores, k=tf.minimum(self.num_retrieved, tf.shape(scores)[-1])
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
        actual_num_retrieved = tf.shape(scores)[-1]

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
            predictions, [batch_size, actual_num_retrieved, tf.shape(predictions)[-1]]
        )

        weights = tf.nn.softmax(scores, axis=-1)

        weighted_predictions = tf.einsum("bvi,bv->bi", predictions, weights)
        weighted_predictions = self.prediction_network.prediction_from_representation(
            weighted_predictions, training=training
        )
        return weighted_predictions

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    def get_representation_size(self):
        return self.prediction_network.get_representation_size()


# from rl755.models.memory.retrieval import *
# if True:
#     from rl755.models.memory import instances

#     sequence_length = 32
#     key_size = 16
#     history_stride = sequence_length // 2
#     num_retrieved = 4

#     prediction_network = instances.deterministic_transformer_32dm_32di(
#         name="prediction"
#     )
#     prediction_network(tf.keras.Input([None, 32 + 15]))

#     query_network = instances.deterministic_transformer_32dm_32di(name="query")
#     query_network(tf.keras.Input([None, 32 + 15]))

#     key_network = instances.deterministic_transformer_32dm_32di(name="key")
#     key_network(tf.keras.Input([None, 32 + 15]))

#     model = EpisodicRetriever(
#         prediction_network=prediction_network,
#         key_network=key_network,
#         query_network=query_network,
#         key_size=key_size,
#         history_stride=history_stride,
#         num_retrieved=num_retrieved,
#     )

#     @tf.function
#     def fn():
#         inputs = tf.random.normal([2, 32, 32 + 15])
#         history = tf.random.normal([2, 512, 32 + 15])
#         history_length = tf.constant([[311], [432]], dtype=tf.int32)

#         full_input = {
#             "inputs": inputs,
#             "history": history,
#             "history_length": history_length,
#         }

#         outputs = model(full_input, training=True)
#         return outputs

#     out = fn()
# # if True:
# #     from rl755.data.envs.caveflyer import EncodedRolloutsVae32d
# #     from itertools import islice

# #     dsb = EncodedRolloutsVae32d()
# #     ds = dsb.get_autoregressive_slices_with_full_history(
# #         sequence_length=32, history_size=1024
# #     )
# #     for x in islice(ds.as_numpy_iterator(), 10):
# #         print(x)
