"""Various controllers."""
import pickle

import numpy as np
from scipy import special
import tensorflow as tf

from .interface import Controller


def _sample_from_logits(logits):
    # TODO: Add docs
    probs = special.softmax(logits, axis=-1)
    cum_prob = np.cumsum(probs, axis=-1)
    r = np.random.uniform(size=cum_prob.shape + (1,))
    return np.argmax(cum_prob > r, axis=-1)


class LinearController(Controller):
    @staticmethod
    def from_flat_arrays(array, in_size, out_size):
        array = np.array(array)
        w, b = array[:, :-out_size], array[:, -out_size:]
        w = np.reshape(w, [-1, out_size, in_size])
        return LinearController(w=w, b=b)

    @staticmethod
    def get_parameter_count(in_size, out_size):
        return in_size * out_size + out_size

    @staticmethod
    def deserialize(bytes_str):
        w, b = pickle.loads(bytes_str)
        return LinearController(w=w, b=b)

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self._out_size = b.shape[-1]
        self._in_size = w.shape[-1]

    def sample_action(self, inputs):
        if isinstance(inputs, tf.Tensor):
            inputs = inputs.numpy()
        logits = np.einsum("...jk,...k->...j", self.w, inputs) + self.b
        logits = np.reshape(logits, [-1, self._out_size])
        return _sample_from_logits(logits)

    def in_size(self):
        return self._in_size

    def out_size(self):
        return self._out_size

    def serialize(self):
        return pickle.dumps((self.w, self.b))
