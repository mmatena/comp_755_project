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
    # The existing implementation seems to be tailored to using the mask only to handle the
    # case where sequences require padding due to different lengths.
    raise ValueError(
        "Looking at the implementation in the library, this looks like it won't work."
    )
    arange = tf.range(seq_len)
    # Looks like the mask shape corresponds to [batch, from_sequence, to_sequence].
    mask = tf.less(tf.expand_dims(arange, axis=0), tf.expand_dims(arange, axis=1))
    mask = tf.cast(tf.expand_dims(mask, axis=0), tf.float32)
    return mask


def _our_create_attention_mask(from_shape, input_mask):
    """Overide of AttentionLayer.create_attention_mask for our purposes.

    In the original version, the input_mask only represents non-padded regions of inputs. We need a
    different shape of the input mask to make the transformer autoregressive. We create the attention
    mask outside of this function and just pass it through.
    """
    print("@@@@@: Yay! I got called!")
    return input_mask


class AutoregressiveTransformer(tf.keras.Model):
    def __init__(self, transformer_params, **kwargs):
        # TODO(mmatena): Add docs
        super().__init__(**kwargs)
        self.transformer_params = transformer_params

    def build(self, input_spec):
        self.transformer = TransformerEncoderLayer.from_params(
            self.transformer_params, name="transformer"
        )
        self.transformer.build(input_spec)

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
        with mock.patch.object(
            AttentionLayer, "create_attention_mask", _our_create_attention_mask
        ):
            output = self.transformer(inputs, mask=mask, training=training)
        return output


# Random testing code!
model = AutoregressiveTransformer(
    TransformerEncoderLayer.Params(
        hidden_size=15,
        num_heads=5,
        num_layers=2,
        intermediate_size=8,
        hidden_dropout=0.1,
    )
)

shape = [16, 8, 2]
model.build(shape)
out = model(tf.ones(shape))
