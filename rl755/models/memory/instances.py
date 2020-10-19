"""Specific instantiations of memory models."""

from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

from . import transformer


def deterministic_transformer_256dm_32di():
    """Returns a deterministic transformer with a model size of 256 and input size of 32."""
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


def deterministic_transformer_64dm_32di():
    """Returns a deterministic transformer with a model size of 64 and input size of 32."""
    output_size = 32  # This must equal the size of the representation.
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
    return transformer.ArTransformer(transformer_params, output_size=output_size)


def deterministic_transformer_32dm_32di():
    """Returns a deterministic transformer with a model size of 32 and input size of 32."""
    output_size = 32  # This must equal the size of the representation.
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
    return transformer.ArTransformer(transformer_params, output_size=output_size)
