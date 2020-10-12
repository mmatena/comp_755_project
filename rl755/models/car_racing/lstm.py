"""Car racing specific lstm code."""
import tensorflow as tf

from rl755.models.common import lstm

def base_lstm():
    output_size = 32  # This must equal the size of the representation.
    hidden_size = 256
    return lstm.LSTM(hidden_size, output_size)