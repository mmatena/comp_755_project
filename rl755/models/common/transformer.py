"""General purpose transformer code.

Note that we will be training on sequences of continuous inputs instead of
discrete tokens, so we'll assume that we aren't using embeddings layers
unless explicitly stated otherwise.
"""
import functools

from bert.attention import AttentionLayer
from bert.transformer import TransformerEncoderLayer

import tensorflow as tf
from unittest import mock

from rl755.models.common.sequential import SequentialModel


def _create_ar_mask(seq_len):
    arange = tf.range(seq_len)
    # Looks like the mask shape corresponds to [batch, from_sequence, to_sequence].
    mask = tf.less_equal(tf.expand_dims(arange, axis=0), tf.expand_dims(arange, axis=1))
    mask = tf.cast(tf.expand_dims(mask, axis=0), tf.float32)
    return mask


def _our_create_attention_mask(from_shape, input_mask, mask):
    """Overide of AttentionLayer.create_attention_mask for our purposes.

    We create the attention mask outside of this function and just pass it through.
    """
    del from_shape, input_mask
    return mask


class ArTransformer(SequentialModel):
    def __init__(self, transformer_params, output_size, **kwargs):
        super().__init__(**kwargs)
        self.transformer_params = transformer_params
        self.output_size = output_size

    def build(self, input_shape):
        hidden_size = self.transformer_params.hidden_size

        self.pos_embeddings = self.add_weight(
            shape=[input_shape[-2], hidden_size],
            initializer="random_normal",
            trainable=True,
        )

        self.initial_layer = tf.keras.layers.Dense(units=hidden_size, activation=None)
        self.initial_layer.build(input_shape)

        self.transformer = TransformerEncoderLayer.from_params(
            self.transformer_params, name="transformer"
        )
        transformer_input_shape = list(input_shape[:-1]) + [hidden_size]
        self.transformer.build(transformer_input_shape)

        self.final_layer = tf.keras.layers.Dense(
            units=self.output_size, activation=None
        )
        self.final_layer.build(list(input_shape[:-1]) + [hidden_size])
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        orig_mask = mask
        seqlen = tf.shape(inputs)[1]
        ar_mask = _create_ar_mask(seqlen)
        if mask is None:
            mask = ar_mask
        else:
            # Here we are assuming mask has shape [batch, seq_len] and is used for padding.
            mask = tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
            mask *= ar_mask

        inputs = self.initial_layer(inputs, training=training)
        inputs += self.pos_embeddings
        # Have to do this hack as the mask in the original transformer just represents non-padded
        # regions of the input. We need a different shape of the input mask to make the transformer
        # autoregressive. The function `_our_create_attention_mask` justs passes through our mask
        # unchanged.
        with mock.patch.object(
            AttentionLayer,
            "create_attention_mask",
            functools.partial(_our_create_attention_mask, mask=mask),
        ):
            output = self.transformer(inputs, mask=orig_mask, training=training)
        self.penultimate_activations = output
        output = self.final_layer(output, training=training)
        return output

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    def get_hidden_representation(self, x, mask=None, training=None, position=-1):
        """Returns the hidden representation used to represent the positiion in the sequence.

        Args:
            x: a tf.Tensor with dtype float32 and shape [batch, sequence, model_input]
            mask: an optional boolean or int tf.Tensor with shape [batch, sequence]
            training: bool, whether to call the transformer in train mode
            position: a positive integer, the index in the sequence hidden representations to
                returns
        Returns:
            A tf.Tensor of dtype float32 and shape [batch, hidden]
        """
        self(x, mask=mask, training=training)
        # TODO(mmatena): This isn't the best layer to use, I'm just doing now for testing
        # as its quick and easy.
        print("TODO(mmatena): This isn't the best layer to use.")
        return self.penultimate_activations[..., position, :]



def base_deterministic_transformer():
    """Returns a deterministic transformer in the base size."""
    output_size = 32  # This must equal the size of the representation.
    num_attention_heads = 4
    hidden_size = 256
    transformer_params = TransformerEncoderLayer.Params(
        num_layers=6,
        hidden_size=hidden_size,
        hidden_dropout=0.1,
        intermediate_size=4 * hidden_size,
        intermediate_activation="gelu",
        num_heads=num_attention_heads,
        size_per_head=int(hidden_size / num_attention_heads),
    )
    return transformer.ArTransformer(transformer_params, output_size=output_size)

# Stuff below this line likely needs to be refactored.
##############################################################################################

# class AutoregressiveLookupTransformer(tf.keras.Model):
#     def __init__(self, ar_transformer, knn_lookup, lambda_knn, **kwargs):
#         super().__init__(**kwargs)
#         assert ar_transformer.return_layer_outputs
#         self.knn_lookup = knn_lookup
#         self.lambda_knn = lambda_knn
#         self.ar_transformer = ar_transformer
#         self.lookup_layer = KnnLookupLayer(self.knn_lookup)

#     def build(self, input_shape):
#         self.ar_transformer.build(input_shape)
#         self.lookup_layer.build(input_shape)
#         super().build(input_shape)

#     def get_queries(self, layers_with_output):
#         layer = self.ar_transformer.transformer.encoder_layers[-1].self_attention_layer
#         return get_output_of_layer(layers_with_output, layer)

#     def load_ar_weights(self, weights_path):
#         self.ar_transformer.load_weights(weights_path)

#     def call(self, inputs, mask=None, training=None):
#         outputs, layers_with_output = self.ar_transformer(
#             inputs, mask=mask, training=training
#         )
#         queries = self.get_queries(layers_with_output)
#         values, distances = self.lookup_layer(queries)

#         # Need to explicitly set the shapes or else the mod
#         values = tf.reshape(
#             values,
#             tf.concat(
#                 [tf.shape(outputs)[:2], [self.knn_lookup.k, tf.shape(outputs)[-1]]],
#                 axis=0,
#             ),
#         )
#         distances = tf.reshape(
#             distances, tf.concat([tf.shape(outputs)[:2], [self.knn_lookup.k]], axis=0)
#         )

#         weights = tf.nn.softmax(-distances)
#         knn_estimates = tf.reduce_sum(
#             values * tf.expand_dims(weights, axis=-1), axis=-2
#         )
#         return self.lambda_knn * knn_estimates + (1 - self.lambda_knn) * outputs
