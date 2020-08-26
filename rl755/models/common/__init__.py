"""General purpose transformer code.

Note that we will be training on sequences of continuous inputs instead of
discrete tokens, so we'll assume that we aren't using embeddings layers
unless explicitly stated otherwise.
"""
from bert.transformer import TransformerEncoderLayer


# Notes: You can call TransformerEncoderLayer with the `mask` kwarg to get an autoregressive model.
