"""General purpose transformer code.

Note that we will be training on sequences of continuous inputs instead of
discrete tokens, so we'll assume that we aren't using embeddings layers
unless explicitly stated otherwise.
"""
import functools

from bert.attention import AttentionLayer
from bert.transformer import TransformerSelfAttentionLayer, TransformerEncoderLayer

import tensorflow as tf
from unittest import mock

from .interface import MemoryComponent


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


class _RepresentationCatcher(tf.keras.layers.Layer):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.representations = {}
        self.key = "default"

    def set_key(self, key):
        self.key = key

    def build(self, *args, **kwargs):
        self.layer.build(*args, **kwargs)
        super().build(*args, **kwargs)

    def call(self, *args, **kwargs):
        output = self.layer(*args, **kwargs)
        self.representations[self.key] = output
        return output


_original_from_params = TransformerSelfAttentionLayer.from_params


def _create_representation_catcher(*args, **kwargs):
    layer = _original_from_params(*args, **kwargs)
    return _RepresentationCatcher(layer)


class ArTransformer(MemoryComponent):
    def __init__(self, transformer_params, output_size, max_sequence_length, **kwargs):
        super().__init__(**kwargs)
        self._has_custom_name = "name" in kwargs
        self.transformer_params = transformer_params
        self.output_size = output_size
        self.max_sequence_length = max_sequence_length

        pos_embeddings_name = "position_embeddings"
        if self._has_custom_name:
            pos_embeddings_name = f"{kwargs['name']}/{pos_embeddings_name}"
        self.pos_embeddings = self.add_weight(
            shape=[max_sequence_length, transformer_params.hidden_size],
            initializer="random_normal",
            trainable=True,
            name=pos_embeddings_name,
        )

    def build(self, input_shape):
        hidden_size = self.transformer_params.hidden_size

        with tf.name_scope("initial"):
            self.initial_layer = tf.keras.layers.Dense(
                units=self.transformer_params.hidden_size,
                activation=None,
                name="initial_dense",
            )
            self.initial_layer.build(input_shape)

        # Not really sure if this is needed here, but putting this here out of caution.
        with mock.patch.object(
            TransformerSelfAttentionLayer,
            "from_params",
            _create_representation_catcher,
        ):
            self.transformer = TransformerEncoderLayer.from_params(
                self.transformer_params,
                name=self.name + "_transformer"
                if self._has_custom_name
                else "transformer",
            )
            transformer_input_shape = list(input_shape[:-1]) + [hidden_size]
            self.transformer.build(transformer_input_shape)

        with tf.name_scope("final"):
            self.final_layer = tf.keras.layers.Dense(
                units=self.output_size, activation=None, name="final_dense"
            )
            self.final_layer.build(list(input_shape[:-1]) + [hidden_size])
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None, pos_embeddings=None):
        orig_mask = mask
        seqlen = tf.shape(inputs)[-2]
        ar_mask = _create_ar_mask(seqlen)
        if mask is None:
            mask = ar_mask
        else:
            # Here we are assuming mask has shape [batch, seq_len] and is used for padding.
            mask = tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
            mask *= ar_mask

        inputs = self.initial_layer(inputs, training=training)
        if pos_embeddings is None:
            pos_embeddings = self.pos_embeddings
        inputs += pos_embeddings[..., :seqlen, :]
        with mock.patch.object(
            TransformerSelfAttentionLayer,
            "from_params",
            _create_representation_catcher,
        ):
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
        output = self.final_layer(output, training=training)
        return output

    def get_loss_fn(self):
        """Train using a MSE loss."""
        return tf.keras.losses.MeanSquaredError()

    @tf.function
    def get_hidden_representation(
        self,
        x,
        mask=None,
        training=None,
        position=-1,
        key="default",
        pos_embeddings=None,
    ):
        """Returns the hidden representation used to represent the positiion in the sequence.

        Following https://openreview.net/pdf?id=HklBjCEKvH, we use the input to the final
        feedwork layer (equivalently the output of the final self-attention layer) for our
        representations.

        Args:
            x: a tf.Tensor with dtype float32 and shape [batch, sequence, model_input]
            mask: an optional boolean or int tf.Tensor with shape [batch, sequence]
            training: bool, whether to call the transformer in train mode
            position: a positive integer, the index in the sequence hidden representations to
                returns
        Returns:
            A tf.Tensor of dtype float32 and shape [batch, hidden]
        """
        representation_layer = self.transformer.encoder_layers[-1].self_attention_layer
        representation_layer.set_key(key)
        self(x, mask=mask, training=training, pos_embeddings=pos_embeddings)
        representation = representation_layer.representations[key]
        return representation[..., position, :]

    def prediction_from_representation(self, representation, training=None):
        # TODO: Add docs, basically go from a represenation to an autoregressive prediction.

        # Fake a sequence dimension.
        representation = tf.expand_dims(representation, axis=-2)

        # Mimic the remainder of the processing within the transformer.
        last_layer = self.transformer.encoder_layers[-1]
        intermediate_output = last_layer.intermediate_layer(representation)
        layer_output = last_layer.output_projector(
            [intermediate_output, representation]
        )

        prediction = self.final_layer(layer_output, training=training)
        # Remove our fake sequence dimension.
        return tf.squeeze(prediction, axis=-2)

    def get_representation_size(self):
        return self.transformer_params.hidden_size
