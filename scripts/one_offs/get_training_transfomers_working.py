import functools
import os

from absl import app

from absl import flags

from bert import BertModelLayer
from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

from rl755.data.car_racing import encoded_rollouts
from rl755.data.car_racing import processing
from rl755.models.car_racing import transformer
from rl755.models.common import transformer as common_transformer

from bert.attention import AttentionLayer

from unittest import mock

input_size = 32
hidden_size = 256
num_attention_heads = 4
transformer_params = TransformerEncoderLayer.Params(
    num_layers=3,
    hidden_size=hidden_size,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    intermediate_size=4 * hidden_size,
    intermediate_activation="gelu",
    num_heads=num_attention_heads,
    size_per_head=int(hidden_size / num_attention_heads),
)

seqlen = 2


class PosEmbeddings(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embeddings = self.add_weight(
            shape=[seqlen, hidden_size],
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.embeddings


layer = TransformerEncoderLayer.from_params(transformer_params, name="transformer")
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(hidden_size, activation=None),
        PosEmbeddings(),
        layer,
        tf.keras.layers.Dense(input_size, activation=None),
    ]
)


def gen():
    while True:
        # x = tf.random.normal([seqlen, input_size])
        # yield x, x
        x = tf.random.normal([seqlen + 1, input_size])
        yield x[:-1], x[1:]  # Shouldn't predict this when AR.


ds = tf.data.Dataset.from_generator(
    gen,
    (tf.float32, tf.float32),
    (tf.TensorShape([seqlen, input_size]), tf.TensorShape([seqlen, input_size])),
)

# ds = encoded_rollouts.random_rollout_slices(seqlen)
# ds = ds.map(lambda x: (x["observations"], x["observations"]))

ds = ds.batch(32)

train_steps = 4000
ds = ds.take(train_steps)


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


with mock.patch.object(
    AttentionLayer,
    "create_attention_mask",
    functools.partial(_our_create_attention_mask, mask=_create_ar_mask(seqlen)),
):

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        # optimizer="adam",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-9
        ),
    )
    model.fit(
        ds,
        epochs=1,
        steps_per_epoch=train_steps,
    )


"""
num_layers     = None
out_layer_ndxs = None   # [-1]

intermediate_size       = None
intermediate_activation = "gelu"

hidden_size        = None
hidden_dropout     = 0.1
initializer_range  = 0.02
adapter_size       = None       # bottleneck size of the adapter - arXiv:1902.00751
adapter_activation = "gelu"
adapter_init_scale = 1e-3

initializer_range = 0.02

hidden_size         = None
num_heads           = None
hidden_dropout      = None
attention_dropout   = 0.1
initializer_range   = 0.02

num_heads         = None
size_per_head     = None
initializer_range = 0.02
query_activation  = None
key_activation    = None
value_activation  = None
attention_dropout = 0.1
negative_infinity = -10000.0  # used for attention scores before softmax
"""
