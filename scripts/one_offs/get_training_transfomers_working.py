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

hidden_size = 4
num_attention_heads = 1
transformer_params = TransformerEncoderLayer.Params(
    num_layers=3,
    hidden_size=hidden_size,
    hidden_dropout=0.1,
    intermediate_size=4 * hidden_size,
    intermediate_activation="gelu",
    num_heads=num_attention_heads,
    size_per_head=int(hidden_size / num_attention_heads),
)

layer = TransformerEncoderLayer.from_params(transformer_params, name="transformer")
model = tf.keras.models.Sequential(
    [
        layer,
        # tf.keras.layers.Dense(hidden_size, activation=None)
    ]
)

seqlen = 2


def gen():
    while True:
        x = tf.random.normal([seqlen, hidden_size])
        yield x, x


ds = tf.data.Dataset.from_generator(
    gen,
    (tf.float32, tf.float32),
    (tf.TensorShape([seqlen, hidden_size]), tf.TensorShape([seqlen, hidden_size])),
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
