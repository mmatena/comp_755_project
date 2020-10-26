"""Specific instantiations of memory models."""

from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

from . import lstm
from . import null
from . import transformer

###############################################################################
# Transformers
###############################################################################


def deterministic_transformer_256dm_32di():
    """Returns a deterministic transformer with a model size of 256 and input size of 32."""
    max_sequence_length = 32
    output_size = 32
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
    return transformer.ArTransformer(
        transformer_params,
        output_size=output_size,
        max_sequence_length=max_sequence_length,
    )


def deterministic_transformer_64dm_32di():
    """Returns a deterministic transformer with a model size of 64 and input size of 32."""
    max_sequence_length = 32
    output_size = 32
    num_attention_heads = 2
    hidden_size = 64
    transformer_params = TransformerEncoderLayer.Params(
        num_layers=6,
        hidden_size=hidden_size,
        hidden_dropout=0.1,
        intermediate_size=4 * hidden_size,
        intermediate_activation="gelu",
        num_heads=num_attention_heads,
        size_per_head=int(hidden_size / num_attention_heads),
    )
    return transformer.ArTransformer(
        transformer_params,
        output_size=output_size,
        max_sequence_length=max_sequence_length,
    )


def deterministic_transformer_32dm_32di():
    """Returns a deterministic transformer with a model size of 32 and input size of 32."""
    max_sequence_length = 32
    output_size = 32
    num_attention_heads = 2
    hidden_size = 32
    transformer_params = TransformerEncoderLayer.Params(
        num_layers=6,
        hidden_size=hidden_size,
        hidden_dropout=0.1,
        intermediate_size=4 * hidden_size,
        intermediate_activation="gelu",
        num_heads=num_attention_heads,
        size_per_head=int(hidden_size / num_attention_heads),
    )
    return transformer.ArTransformer(
        transformer_params,
        output_size=output_size,
        max_sequence_length=max_sequence_length,
    )


###############################################################################
# LSTMs
###############################################################################


def deterministic_lstm_256dm_32di():
    """Returns a deterministic lstm with a model size of 256 and input size of 32."""
    return lstm.Lstm(hidden_size=256, output_size=32)


def deterministic_lstm_64dm_32di():
    """Returns a deterministic lstm with a model size of 64 and input size of 32."""
    return lstm.Lstm(hidden_size=64, output_size=32)


def deterministic_lstm_32dm_32di():
    """Returns a deterministic lstm with a model size of 32 and input size of 32."""
    return lstm.Lstm(hidden_size=32, output_size=32)


###############################################################################
# No memory
###############################################################################


def no_mem():
    return null.NullMemory()
