"""General purpose transformer code.

Note that we will be training on sequences of continuous inputs instead of
discrete tokens, so we'll assume that we aren't using embeddings layers
unless explicitly stated otherwise.
"""
import collections
import functools

from bert.attention import AttentionLayer
from bert.transformer import TransformerEncoderLayer
from unittest import mock
import tensorflow as tf
import tensorflow_probability as tfp

from rl755.models.car_racing.knn import KnnLookupLayer
from rl755.models.common.layers import MixtureOfGaussiansLayer

tfd = tfp.distributions

LayerWithOutput = collections.namedtuple("LayerWithOutput", ["layer", "output"])


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


_original_layer_call = tf.keras.layers.Layer.__call__


def _get_our_layer_call(array):
    def fn(self, *args, **kwargs):
        output = _original_layer_call(self, *args, **kwargs)
        array.append(LayerWithOutput(layer=self, output=output))
        return output

    return fn


class AutoregressiveTransformer(tf.keras.Model):
    def __init__(
        self,
        transformer_params,
        output_size,
        num_components,
        return_layer_outputs=False,
        **kwargs
    ):
        # TODO(mmatena): Add docs
        super().__init__(**kwargs)
        self.transformer_params = transformer_params
        self.output_size = output_size
        self.num_components = num_components
        self.return_layer_outputs = return_layer_outputs
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.logits = self.add_weight(
            shape=[num_components], initializer="random_normal", trainable=True
        )

    def build(self, input_shape):
        hidden_size = self.transformer_params.hidden_size
        self.initial_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=hidden_size, activation=None)
        )
        self.initial_layer.build(input_shape)

        self.transformer = TransformerEncoderLayer.from_params(
            self.transformer_params, name="transformer"
        )
        transformer_input_shape = list(input_shape[:-1]) + [hidden_size]
        self.transformer.build(transformer_input_shape)

        final_layer_size = 2 * self.num_components * self.output_size
        self.final_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=final_layer_size, activation=None)
        )
        self.final_layer.build(list(input_shape[:-1]) + [hidden_size])
        super().build(input_shape)

    def call(self, inputs, mask=None, training=None):
        call_inner = lambda: self._call_inner(inputs, mask=mask, training=training)
        if not self.return_layer_outputs:
            return call_inner()
        layers_with_output = []
        override = _get_our_layer_call(layers_with_output)
        with mock.patch.object(tf.keras.layers.Layer, "__call__", override):
            outputs = call_inner()
            return outputs, layers_with_output

    def _call_inner(self, inputs, mask=None, training=None):
        # TODO(mmatena): Make sure this is right.
        orig_mask = mask
        seqlen = tf.shape(inputs)[1]
        ar_mask = _create_ar_mask(seqlen)
        if mask is None:
            mask = ar_mask
        else:
            # Here we are assuming mask has shape [batch, seq_len] and is used for padding.
            mask = tf.cast(tf.expand_dims(mask, axis=1), tf.float32)
            mask *= ar_mask

        inputs = self.initial_layer(inputs, training=training)
        # Have to do this hack as the mask in the original transformer just represents non-padded
        # regions of the input. We need a different shape of the input mask to make the transformer
        # autoregressive. The function `_our_create_attention_mask` justs passes through our mask
        # unchanged.
        with mock.patch.object(
            AttentionLayer,
            "create_attention_mask",
            functools.partial(_our_create_attention_mask, mask=mask),
        ):
            output = self.transformer(inputs, mask=orig_mask, training=training)
        output = self.final_layer(output, training=training)
        return output

    def _get_gauss_params(self, outputs):
        locs, scales = tf.split(outputs, num_or_size_splits=2, axis=-1)
        locs = tf.split(
            locs,
            num_or_size_splits=self.num_components * [self.output_size],
            axis=-1,
        )
        scales = tf.split(
            tf.nn.softplus(scales),
            num_or_size_splits=self.num_components * [self.output_size],
            axis=-1,
        )
        return locs, scales

    def _get_gauss_components(self, outputs):
        locs, scales = self._get_gauss_params(outputs)
        return [
            tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)
            for loc, scale in zip(locs, scales)
        ]

    def get_mix_of_gauss(self, outputs):
        batch_dims = tf.shape(outputs)[:-1]
        logits = tf.reshape(
            self.logits, (len(outputs.shape) - 1) * [1] + [self.num_components]
        )
        logits = tf.broadcast_to(
            logits, tf.concat([batch_dims, [self.num_components]], axis=0)
        )
        cat_dist = tfd.Categorical(logits=logits)
        return tfd.Mixture(cat=cat_dist, components=self._get_gauss_components(outputs))

    def nll_loss(self, global_batch_size=None):
        def nll_loss(y_true, y_pred):
            y_pred = self.get_mix_of_gauss(y_pred)
            loss = -y_pred.log_prob(y_true)
            if not global_batch_size:
                loss = tf.reduce_mean(loss)
            else:
                loss = tf.nn.compute_average_loss(
                    loss, global_batch_size=global_batch_size
                )
            tf.summary.scalar("loss", data=loss, step=self.step)
            return loss

        return nll_loss

    # def _train_step_inner(self, data):
    #     self.step.assign_add(1)

    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         out_dist = self(x, training=True)
    #         loss = -out_dist.log_prob(y)

    #     trainable_vars = self.trainable_variables
    #     gradients = tape.gradient(loss, trainable_vars)
    #     self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #     tf.summary.scalar("loss", data=loss, step=self.step)
    #     return {"loss": loss}

    # def train_step(self, data):
    #     if self.mirrored_strategy:
    #         pass
    #     else:
    #         return self._train_step_inner(data)


# TODO(mmatena): Reduce code duplication here.
def get_output_of_layer(layers_with_output, layer_):
    for layer, output in layers_with_output:
        if layer is layer_:
            return output
    raise ValueError("Layer not found.")


class AutoregressiveLookupTransformer(tf.keras.Model):
    def __init__(self, ar_transformer, knn_lookup, lambda_knn, **kwargs):
        super().__init__(**kwargs)
        assert ar_transformer.return_layer_outputs
        self.knn_lookup = knn_lookup
        self.lambda_knn = lambda_knn
        self.ar_transformer = ar_transformer
        self.lookup_layer = KnnLookupLayer(self.knn_lookup)

    def build(self, input_shape):
        self.ar_transformer.build(input_shape)
        self.lookup_layer.build(input_shape)
        super().build(input_shape)

    def get_queries(self, layers_with_output):
        layer = self.ar_transformer.transformer.encoder_layers[-1].self_attention_layer
        return get_output_of_layer(layers_with_output, layer)

    def load_ar_weights(self, weights_path):
        self.ar_transformer.load_weights(weights_path)

    def call(self, inputs, mask=None, training=None):
        outputs, layers_with_output = self.ar_transformer(
            inputs, mask=mask, training=training
        )
        queries = self.get_queries(layers_with_output)
        values, distances = self.lookup_layer(queries)

        # Need to explicitly set the shapes or else the mod
        values = tf.reshape(
            values,
            tf.concat(
                [tf.shape(outputs)[:2], [self.knn_lookup.k, tf.shape(outputs)[-1]]],
                axis=0,
            ),
        )
        distances = tf.reshape(
            distances, tf.concat([tf.shape(outputs)[:2], [self.knn_lookup.k]], axis=0)
        )

        weights = tf.nn.softmax(-distances)
        knn_estimates = tf.reduce_sum(
            values * tf.expand_dims(weights, axis=-1), axis=-2
        )
        return self.lambda_knn * knn_estimates + (1 - self.lambda_knn) * outputs
