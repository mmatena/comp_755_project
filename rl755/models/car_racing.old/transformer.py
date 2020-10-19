"""Car racing specific transformer code."""
from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

from rl755.models.common import transformer


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


def deterministic_transformer_64():
    """Returns a deterministic transformer with a model size of 64."""
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


def deterministic_transformer_32():
    """Returns a deterministic transformer with a model size of 32."""
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
