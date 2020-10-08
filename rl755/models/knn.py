"""Code for looking up the k-nearest neighbors.
See https://arxiv.org/pdf/1911.00172.pdf for more details.
"""
import numpy as np
import tensorflow as tf

from rl755.common import structs

from config.config import get_knn_tfrecords_pattern



# TODO(mmatena): Change when we have something permanent.
TFRECORDS_PATTERN = "/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/encoded_knn/knn_data.tfrecord*"


def _get_knn_ds(num_points=None):
    files = tf.io.matching_files(TFRECORDS_PATTERN)

    files = tf.data.Dataset.from_tensor_slices(files)
    ds = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    if num_points is not None:
        ds = ds.take(num_points)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds.map(
        structs.tfrecord_to_key_value, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


def _ds_to_np(ds):
    max_elems = np.iinfo(np.int32).max
    ds = ds.batch(max_elems)
    return tf.data.experimental.get_single_element(ds)


def _get_knn_np(num_points=None):
    knn_ds = _get_knn_ds(num_points=num_points)
    knn_ds = _ds_to_np(knn_ds)

    return knn_ds["key"], knn_ds["value"]


def _create_searcher(array, k):
    # TODO(mmatena): I can't figure out how to do the LD_LIBRARY and launch python
    # in one command without not giving python access to the GPUS for some reason.
    import scann  # noqa: E402

    size = array.shape[0]
    # TODO(mmatena): Most of these value are copied from a demo. Look into them more.
    return (
        scann.ScannBuilder(array, num_neighbors=k, distance_measure="dot_product")
        .tree(
            num_leaves=int(np.sqrt(size)),
            num_leaves_to_search=100,
            training_sample_size=250000,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(k ** 2)
        .create_pybind()
    )


class KnnLookup(object):
    def __init__(self, k, num_points=None):
        self.k = k
        keys, values = _get_knn_np(num_points=num_points)
        self.searcher = _create_searcher(keys, k=k)
        self.values = values

    def _get_batched(self, queries, **kwargs):
        # TODO(mmatena): add docs
        # Returned shapes = [<batch>, k, d_value], [<batch>, k]
        batch_shape, key_size = tf.shape(queries)[:-1], tf.shape(queries)[-1]
        queries = tf.reshape(queries, [-1, key_size])

        neighbors, distances = self.searcher.search_batched(queries.numpy(), **kwargs)
        neighbors = tf.cast(neighbors, tf.int32)
        values = tf.gather(self.values, neighbors, axis=0)

        value_size = tf.shape(values)[-1]
        values = tf.reshape(
            values, tf.concat([batch_shape, [self.k, value_size]], axis=0)
        )
        distances = tf.reshape(distances, tf.concat([batch_shape, [self.k]], axis=0))

        return values, distances

    def get_batched(self, queries):
        # TODO(mmatena): add support for kwargs for search_batched
        return tf.py_function(
            self._get_batched,
            inp=[queries],
            Tout=[tf.float32, tf.float32],
        )


class KnnLookupLayer(tf.keras.layers.Layer):
    def __init__(self, knn_lookup, **kwargs):
        super().__init__(**kwargs)
        self.knn_lookup = knn_lookup

    def call(self, inputs):
        return self.knn_lookup.get_batched(inputs)
