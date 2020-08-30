"""General purpose transformer code.

Note that we will be training on sequences of continuous inputs instead of
discrete tokens, so we'll assume that we aren't using embeddings layers
unless explicitly stated otherwise.
"""
from bert.attention import AttentionLayer
from bert.transformer import TransformerEncoderLayer
from unittest import mock
import tensorflow as tf


def _create_ar_mask(seq_len):
    arange = tf.range(seq_len)
    # Looks like the mask shape corresponds to [batch, from_sequence, to_sequence].
    mask = tf.less_equal(tf.expand_dims(arange, axis=0), tf.expand_dims(arange, axis=1))
    mask = tf.cast(tf.expand_dims(mask, axis=0), tf.float32)
    return mask


def _our_create_attention_mask(from_shape, input_mask):
    """Overide of AttentionLayer.create_attention_mask for our purposes.

    We create the attention mask outside of this function and just pass it through.
    """
    return input_mask


class AutoregressiveTransformer(tf.keras.Model):
    def __init__(self, transformer_params, output_size, **kwargs):
        # TODO(mmatena): Add docs
        super().__init__(**kwargs)
        self.transformer_params = transformer_params
        self.output_size = output_size

    def build(self, input_shape):
        hidden_size = self.transformer_params.hidden_size
        self.initial_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=hidden_size, activation=None)
        )
        self.initial_layer.build(input_shape)
        self.transformer = TransformerEncoderLayer.from_params(
            self.transformer_params, name="transformer"
        )
        transformer_input_shape = list(input_shape[:-1]) + [hidden_size]
        self.transformer.build(transformer_input_shape)
        self.final_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=self.output_size, activation=None)
        )
        self.final_layer.build(list(input_shape[:-1]) + [hidden_size])
        super().build(input_shape)

    @tf.function
    def get_something(self, inputs, mask=None, training=None):
        seqlen = tf.shape(inputs)[1]
        ar_mask = _create_ar_mask(seqlen)
        if mask is None:
            mask = ar_mask
        else:
            # Here we are assuming mask has shape [batch, seq_len] and is used for padding.
            mask = tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
            mask *= ar_mask

        inputs = self.initial_layer(inputs, training=training)
        # Have to do this hack as the mask in the original transformer just represents non-padded
        # regions of the input. We need a different shape of the input mask to make the transformer
        # autoregressive. The function `_our_create_attention_mask` justs passes through our mask
        # unchanged.
        with mock.patch.object(
            AttentionLayer, "create_attention_mask", _our_create_attention_mask
        ):
            self.transformer(inputs, mask=mask, training=training)
        return self.transformer.encoder_layers[-1].self_attention_layer.output

    def call(self, inputs, mask=None, training=None):
        # TODO(mmatena): Make sure this is right.
        seqlen = tf.shape(inputs)[1]
        ar_mask = _create_ar_mask(seqlen)
        if mask is None:
            mask = ar_mask
        else:
            # Here we are assuming mask has shape [batch, seq_len] and is used for padding.
            mask = tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
            mask *= ar_mask

        inputs = self.initial_layer(inputs, training=training)
        # Have to do this hack as the mask in the original transformer just represents non-padded
        # regions of the input. We need a different shape of the input mask to make the transformer
        # autoregressive. The function `_our_create_attention_mask` justs passes through our mask
        # unchanged.
        with mock.patch.object(
            AttentionLayer, "create_attention_mask", _our_create_attention_mask
        ):
            output = self.transformer(inputs, mask=mask, training=training)
        output = self.final_layer(output, training=training)
        return output
