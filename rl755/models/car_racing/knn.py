"""Code for looking up the k-nearest neighbors.
See https://arxiv.org/pdf/1911.00172.pdf for more details.
"""
import tensorflow as tf

from rl755.common import structs

# TODO(mmatena): Change when we have something permanent.
TFRECORDS_PATTERN = (
    "/pine/scr/m/m/mmatena/comp_755_project/test_knn_data/knn_data.tfrecord*"
)


def get_knn_ds(num_points=None):
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


for x in get_knn_ds():
    print(x)
    break
