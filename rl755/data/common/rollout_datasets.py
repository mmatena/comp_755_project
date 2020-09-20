"""Common code for rollout-based datasets."""
import functools

import tensorflow as tf

from rl755.data.common import processing


class RolloutDatasetBuilder(object):
    """The abstract base class for returning a rollout-based dataset."""

    # Methods to override.

    def _environment(self):
        """The environment used to generate the rollouts.

        Returns:
            A rl755.environments.Environments enum.
        """
        raise NotImplementedError()

    def _tfrecords_pattern(self):
        """The glob pattern used that matches all of the tfrecords containing the dataset.

        It should contain a "split" format kwarg. For example:
            "path/to/dir/{split}/data.tfrecord*"

        The split will be replaced with either "train" or "validation" to access the
        train and validation sets, respectively.
        """
        raise NotImplementedError()

    def _stored_observation_dtype(self):
        """The tf.dtypes.DType that the observations are STORED as in the tfrecords.

        Note that this can be different than the actual dtype of the observations
        after they have been processed. Due a quirk of how tfrecords work, we must
        store images as type tf.string. Once they have been read, we can then process
        them to a tf.uint8 or tf.float32 dtype.
        """
        raise NotImplementedError()

    def _observation_shape(self):
        """The shape of each observation.

        Returns:
            A tuple of ints representing the shape of an individual observation.
        """
        raise NotImplementedError()

    def _additional_features(self):
        """Additional features that are stored in the dataset.

        All of the datasets are assumed to contain "observations", "actions", and
        "rewards". This lets you add additional features that are stored in the
        dataset.

        An example usage is would be including the "observation_std_devs" feature
        in a dataset of rollouts that have been encoded by a VAE.

        Returns:
            A dict mapping string keys to tf.io.VarLenFeature(dtype) values, where
            the dtype is the dtype used to store the feature.
        """
        return {}

    def _process_observation(self, observation):
        return observation

    def _process_rollout(self, example):
        return example

    # Public interface.

    def action_size(self):
        action_shape = self._environment().action_shape
        assert len(action_shape) == 1
        return action_shape[0]

    def parse_tfrecord(self, x, process_observations=True):
        features = {
            "observations": tf.io.VarLenFeature(self._stored_observation_dtype()),
            "actions": tf.io.VarLenFeature(tf.float32),
            "rewards": tf.io.VarLenFeature(tf.float32),
        }
        features.update(self._additional_features())
        _, x = tf.io.parse_single_sequence_example(x, sequence_features=features)
        x = {k: tf.sparse.to_dense(v) for k, v in x.items()}
        x["rewards"] = tf.squeeze(x["rewards"])
        if process_observations:
            x["observations"] = self._process_observations(x["observations"])
        x = self._process_rollout(x)
        return x

    def get_tfrecords_pattern(self, split):
        return self._tfrecords_pattern().format(split=split)

    def get_tfrecord_files(self, split):
        return tf.io.matching_files(self.get_tfrecords_pattern(split=split))

    def rollouts_ds(self, split="train", process_observations=True):
        files = self.get_tfrecord_files(split=split)

        files = tf.data.Dataset.from_tensor_slices(files)
        ds = files.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        ds = ds.repeat()
        return ds.map(
            functools.partial(
                self.parse_tfrecord, process_observations=process_observations
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

    def random_rollout_slices_ds(self, slice_size, split="train"):
        def map_fn(x):
            x = processing.slice_example(x, slice_size=slice_size)
            x["observations"] = self._process_observations(x["observations"])
            return x

        ds = self.rollouts_ds(split=split, process_observations=False)
        return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def random_rollout_observations(self, split="train", obs_sampled_per_rollout=100):
        def random_obs(x):
            rollout_length = tf.shape(x["observations"])[0]
            index = tf.random.uniform(
                [obs_sampled_per_rollout], 0, rollout_length, dtype=tf.int32
            )
            observation = tf.gather(x["observations"], index, axis=0)
            observation = self._process_observations(observation)
            return {"observation": observation}

        def set_shape(x):
            return {
                "observation": tf.reshape(x["observation"], self._observation_shape())
            }

        ds = self.rollouts_ds(split=split, process_observations=False)
        ds = ds.map(random_obs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
        ds = ds.map(set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds


class RawImageRolloutDatasetBuilder(RolloutDatasetBuilder):

    # Methods to override.

    def _environment(self):
        raise NotImplementedError()

    def _tfrecords_pattern(self):
        raise NotImplementedError()

    # Private methods.

    def _observation_shape(self):
        return self._environment().observation_shape

    def _stored_observation_dtype(self):
        return tf.string

    def _process_observation(self, observations):
        observations = tf.map_fn(
            functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
            tf.squeeze(observations),
            dtype=tf.uint8,
        )
        # Convert to floats in the range [0, 1]
        observations = tf.cast(observations, tf.float32) / 255.0
        return observations


class EncodedRolloutDatasetBuilder(RolloutDatasetBuilder):

    # Methods to override.

    def _environment(self):
        raise NotImplementedError()

    def _tfrecords_pattern(self):
        raise NotImplementedError()

    def representation_size(self):
        raise NotImplementedError()

    def _stored_observation_dtype(self):
        return tf.float32

    def _sample_observations(self, example):
        return example["observations"]

    # Private methods.

    def _observation_shape(self):
        return (self.representation_size(),)

    # Public interface.

    def get_autoregressive_slices(
        self, sequence_length, sample_observations=False, split="train"
    ):
        rep_size = self.representation_size()
        action_size = self.action_size()

        def map_fn(x):
            a = x["actions"][:, :action_size]
            if sample_observations:
                o = self._sample_observations(x)
            else:
                o = x["observations"]
            inputs = tf.concat(
                [o[:-1], a[:-1]],
                axis=-1,
            )
            targets = o[1:] - o[:-1]
            inputs = tf.reshape(inputs, [sequence_length, rep_size + action_size])
            targets = tf.reshape(targets, [sequence_length, rep_size])
            return inputs, targets

        return self.random_rollout_slices_ds(
            self, slice_size=sequence_length + 1, split=split
        )
