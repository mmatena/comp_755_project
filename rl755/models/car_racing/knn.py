"""Code for looking up the k-nearest neighbors.
See https://arxiv.org/pdf/1911.00172.pdf for more details.
"""
import numpy as np
import scann
import tensorflow as tf

from rl755.common import structs

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
    max_elems = np.iinfo(np.int64).max
    ds = ds.batch(max_elems)
    return tf.data.experimental.get_single_element(ds)


def _get_knn_np(num_points=None):
    knn_ds = _get_knn_ds(num_points=num_points)
    knn_ds = _ds_to_np(knn_ds)

    print("a", knn_ds["value"].shape)  # Just curious.
    return knn_ds["key"], knn_ds["value"]


def _create_searcher(array, k):
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
        keys, values = _get_knn_np(num_points=num_points)
        self.searcher = _create_searcher(keys, k=k)
        self.values = values

    def get_batched(self, queries, **kwargs):
        neighbors, distances = self.searcher.search_batched(queries, **kwargs)
        values = tf.gather(self.values, neighbors, axis=0)
        return values, tf.constant(distances)


# array = _get_knn_np()
# searcher = _create_searcher(array, k=10)
