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

hidden_size = 256
num_attention_heads = 4
transformer_params = TransformerEncoderLayer.Params(
    num_layers=3,
    hidden_size=hidden_size,
    hidden_dropout=0.1,
    attention_dropout=0.1,
    intermediate_size=4 * hidden_size,
    intermediate_activation=None,  # #############################3
    num_heads=num_attention_heads,
    size_per_head=int(hidden_size / num_attention_heads),
)

seqlen = 2

layer = TransformerEncoderLayer.from_params(transformer_params, name="transformer")
model = tf.keras.models.Sequential(
    [
        layer,
        tf.keras.layers.Dense(hidden_size, activation=None),
    ]
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


def gen():
    while True:
        x = tf.random.normal([seqlen, hidden_size])
        # x = tf.random.normal([seqlen * hidden_size])
        yield x, x


ds = tf.data.Dataset.from_generator(
    gen,
    (tf.float32, tf.float32),
    (tf.TensorShape([seqlen, hidden_size]), tf.TensorShape([seqlen, hidden_size])),
    # (tf.TensorShape([seqlen * hidden_size]), tf.TensorShape([seqlen * hidden_size])),
)
ds = ds.batch(32)

train_steps = 5000
ds = ds.take(train_steps)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer="adam",
)
model.fit(
    ds,
    epochs=1,
    steps_per_epoch=train_steps,
)
