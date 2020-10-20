"""Utility functions for dealing with tfrecords."""
import math
import os

import tensorflow as tf

from rl755.common import misc


def to_bytes_feature(tensor):
    # TODO: Add docs.
    tensor_str = tf.io.serialize_tensor(tensor).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_str]))


def to_bytes_feature_list(array):
    # TODO: Add docs.
    features = [
        tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(t).numpy()])
        )
        for t in array
    ]
    return tf.train.FeatureList(feature=features)


def to_float_feature_list(array):
    # TODO: Add docs.
    features = [tf.train.Feature(float_list=tf.train.FloatList(value=t)) for t in array]
    return tf.train.FeatureList(feature=features)


def to_int64_feature_list(array):
    # TODO: Add docs.
    features = [tf.train.Feature(int64_list=tf.train.Int64List(value=t)) for t in array]
    return tf.train.FeatureList(feature=features)


class FixedSizeShardedWriter(object):
    # TODO: Add docs, make usable with "with" statement
    def __init__(self, directory, filename, total_count, desired_shard_mb):
        # TODO: Add docs
        self.directory = directory
        self.filename = filename
        self.total_count = total_count
        self.desired_shard_mb = desired_shard_mb

        self.current_shard_index = 0
        self.current_shard_items = 0

        self.current_shard_writer = None

        self.item_mb = None
        self.items_per_shard = None
        self.shard_count = None

    def _set_item_mb(self, record):
        if self.item_mb is not None:
            return
        self.item_mb = len(record) / 1024 ** 2
        self.items_per_shard = min(1, round(self.desired_shard_mb / self.item_mb))
        self.shard_count = int(math.ceil(self.total_count / self.items_per_shard))

    def _create_shard_writer(self):
        base_name = misc.sharded_filename(
            self.filename,
            shard_index=self.current_shard_index,
            num_shards=self.shard_count,
        )
        file_shard = os.path.join(self.directory, base_name)

        if self.current_shard_writer:
            self.current_shard_writer.close()
        self.current_shard_writer = tf.io.TFRecordWriter(file_shard)

    def _next_shard(self):
        self.current_shard_index += 1
        self.current_shard_items = 0
        self._create_shard_writer()

    def _write_record(self, record):
        if not self.current_shard_writer:
            self._create_shard_writer()

        if self.current_shard_items >= self.items_per_shard:
            self._next_shard()

        self.current_shard_writer.write(record)

        self.current_shard_items += 1

    def write(self, records):
        # TODO: Add docs
        if not isinstance(records, (list, tuple)):
            records = [records]

        for record in records:
            self._set_item_mb(record)
            self._write_record(record)

    def close(self):
        if self.current_shard_writer:
            self.current_shard_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        if exc_type is not None:
            # Just raise the exception. IDK if I'm doing that here.
            raise exc_value
        return True
