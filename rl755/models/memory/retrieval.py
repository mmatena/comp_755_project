"""Models that explicitly retrieve experiences."""
import numpy as np
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

    def get_value_stride(self):
        return self.history_stride

    def get_key_size(self):
        return self.key_size

    # def build(self, input_shape):
    #     # These represent retrieving no documents.
    #     self.empty_key = self.add_weight(
    #         shape=[self.key_size],
    #         initializer="random_normal",
    #         trainable=True,
    #         name="empty_key",
    #     )
    #     self.empty_value = self.add_weight(
    #         shape=input_shape["inputs"][-1:],
    #         initializer="random_normal",
    #         trainable=True,
    #         name="empty_value",
    #     )

    def _get_prediction_pos_embeddings(self):
        # Actually also includes type embeddings.
        pos_embeddings = self.prediction_network.pos_embeddings
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
        if max_history_length < sequence_length:
            return tf.zeros([history.shape[0], 0, sequence_length, history.shape[-1]])
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

        # TODO: Use an empty value/key. Need to move from here since these won't get
        # gradients propagated here.
        # empty_value = tf.broadcast_to(self.empty_value[None, ...], tf.shape(values))
        # values = tf.concat([empty_value, values], axis=1)

        # empty_key = tf.broadcast_to(self.empty_key[None, ...], tf.shape(keys))
        # keys = tf.concat([empty_key, keys], axis=1)

        return keys, values

    def _get_queries(self, inputs, training, position=-1):
        queries = self.query_network.get_hidden_representation(
            inputs, training=training, position=position
        )
        queries = self.query_proj(queries)
        return queries

    def _retrieve(self, inputs, history, history_length, training, position=-1):
        sequence_length = tf.shape(inputs)[-2]

        keys, values = self._get_key_values(
            history,
            history_length=history_length,
            sequence_length=sequence_length,
            training=training,
            position=position,
        )
        queries = self._get_queries(inputs, training, position=position)

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
            retrieved_values, training=training, position=position, rep_key="with_grads"
        )

        retrieved_scores = tf.einsum("bvi,bi->bv", retrieved_keys, queries)
        ignore_mask = tf.cast(
            tf.equal(retrieved_scores_no_grad, _NEG_VALUE), dtype=tf.float32
        )
        retrieved_scores = (
            1.0 - ignore_mask
        ) * retrieved_scores + ignore_mask * _NEG_VALUE

        return retrieved_values, retrieved_scores

    def _extract_values_func(self, history, key_indices, sequence_length):
        sequence_length = sequence_length.numpy()
        starts = (key_indices * self.history_stride).numpy()
        num_retrieved = starts.shape[-1]

        history = history.numpy()

        values = np.empty(
            [history.shape[0], num_retrieved, sequence_length, history.shape[-1]]
        )

        for i in range(num_retrieved):
            values[:, i, :, :] = history[:, starts : starts + sequence_length]

        return tf.constant(values)

    def _extract_values(self, history, key_indices, sequence_length):
        # history.shape = [batch, max_samples, state]
        # key_indices.shape = [batch, num_retrieved]
        starts = key_indices * self.history_stride
        # inds.shape = [batch, num_retrieved, sequence_length]
        inds = starts[..., None] + tf.range(sequence_length)[None, None, :]
        return tf.gather(history, inds, axis=-2, batch_dims=1)

    @tf.function
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
        sequence_length = tf.shape(inputs)[-2]
        batch_size = tf.shape(inputs)[0]

        if num_keys == 0:
            # TODO: Do something better here.
            prediction_inputs = tf.concat(2 * [inputs], axis=-2)
            return self.prediction_network.get_hidden_representation(
                prediction_inputs,
                training=training,
                position=position,
                pos_embeddings=self._get_prediction_pos_embeddings(),
            )

        queries = self._get_queries(inputs, training=training, position=position)
        keys = keys[:, :num_keys]

        scores = tf.einsum("bvi,bi->bv", keys, queries)
        scores, key_indices = tf.math.top_k(
            scores, k=tf.minimum(self.num_retrieved, num_keys)
        )

        values = self._extract_values(history, key_indices, sequence_length)
        actual_num_retrieved = tf.shape(scores)[-1]

        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.broadcast_to(inputs, tf.shape(values))

        prediction_inputs = tf.concat([values, inputs], axis=-2)
        prediction_inputs = tf.reshape(
            prediction_inputs, [-1, 2 * sequence_length, tf.shape(inputs)[-1]]
        )

        representations = self.prediction_network.get_hidden_representation(
            prediction_inputs,
            training=training,
            position=position,
            pos_embeddings=self._get_prediction_pos_embeddings(),
        )
        representations = tf.reshape(
            representations,
            [batch_size, actual_num_retrieved, tf.shape(representations)[-1]],
        )

        weights = tf.nn.softmax(scores, axis=-1)

        weighted_representation = tf.einsum("bvi,bv->bi", representations, weights)
        return weighted_representation

    @tf.function
    def key_from_value(self, value, training=None):
        keys = self.key_network.get_hidden_representation(
            value, training=training, position=-1, key="key_from_value"
        )
        return self.key_proj(keys)

    def get_hidden_representation_train(
        self, inputs, history, history_length, mask=None, training=None, position=-1
    ):
        sequence_length = tf.shape(inputs)[-2]
        batch_size = tf.shape(inputs)[0]

        values, scores = self._retrieve(
            inputs, history, history_length, training=training
        )
        actual_num_retrieved = tf.shape(scores)[-1]

        inputs = tf.expand_dims(inputs, 1)
        inputs = tf.broadcast_to(inputs, tf.shape(values))

        prediction_inputs = tf.concat([values, inputs], axis=-2)
        prediction_inputs = tf.reshape(
            prediction_inputs, [-1, 2 * sequence_length, tf.shape(inputs)[-1]]
        )

        representations = self.prediction_network.get_hidden_representation(
            prediction_inputs,
            training=training,
            position=position,
            pos_embeddings=self._get_prediction_pos_embeddings(),
        )
        representations = tf.reshape(
            representations,
            [batch_size, actual_num_retrieved, tf.shape(representations)[-1]],
        )

        weights = tf.nn.softmax(scores, axis=-1)

        weighted_representation = tf.einsum("bvi,bv->bi", representations, weights)
        return weighted_representation

    def _call(self, inputs, history, history_length, mask=None, training=None):
        weighted_representation = self.get_hidden_representation_train(
            inputs, history, history_length, mask=mask, training=training, position=-1
        )
        weighted_predictions = self.prediction_network.prediction_from_representation(
            weighted_representation, training=training
        )
        return weighted_predictions

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    def get_representation_size(self):
        return self.prediction_network.get_representation_size()


class NoHistoryWrapper(MemoryComponentWithHistory):
    """Lets us use a transformer/LSTM with no history as control against retrieval network. """

    def __init__(self, memory_component, **kwargs):
        super().__init__(self, **kwargs)
        self.memory_component = memory_component

    def _call(self, inputs, history, history_length, mask=None, training=None):
        del history, history_length
        predictions = self.memory_component(inputs, mask=mask, training=training)
        return predictions[..., -1, :]

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
        del history, keys, history_length, num_keys
        return self.memory_component.get_hidden_representation(
            inputs, mask=mask, training=training, position=position
        )

    def get_value_stride(self):
        return int(1e6)

    def get_key_size(self):
        return 1

    def key_from_value(self, value, training=None):
        return tf.zeros([tf.shape(value)[0], self.get_key_size()])

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    def get_representation_size(self):
        return self.memory_component.get_representation_size()
