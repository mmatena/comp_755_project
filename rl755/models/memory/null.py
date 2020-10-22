"""Memory component that does nothing."""
import tensorflow as tf

from .interface import MemoryComponent


class NullMemory(MemoryComponent):
    # TODO: Probably add docs

    # Note: IDK if 0 would work, 1 seems like it'd work.
    representation_size = 1

    def call(self, inputs, mask=None, training=None):
        # Do not call this model.
        raise NotImplementedError()

    def get_loss_fn(self):
        # Do not train this model.
        raise NotImplementedError()

    def get_hidden_representation(self, x, mask=None, training=None, position=-1):
        return tf.expand_dims(tf.zeros(tf.shape(x)[:-2]), axis=-1)

    def get_representation_size(self):
        return self.representation_size
