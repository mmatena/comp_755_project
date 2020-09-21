"""Common code for rollout-based datasets."""
import functools

import tensorflow as tf

from rl755.data.common import processing


class RolloutDatasetBuilder(object):
    """The abstract base class for returning a rollout-based dataset."""

    # Methods to potentially override.

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

    def _process_observations(self, observation):
        """Perform some processing processing on the observation tensor.

        The original purpose of this is that we needed to store images as strings
        due to a quirk in tensorflow. In this case, this function is used to
        parse the string version.

        Args:
            observation: a tf.Tensor with dtype `_stored_observation_dtype()`
        Returns:
            A tf.Tensor.
        """
        return observation

    def _process_rollout(self, example):
        """Processes a full rollout.

        Useful if we need to do some common preprocessing on all examples or
        if we need to parse some features other than "observations", "actions",
        and "rewards".

        This will be done after `_process_observations` has been called.

        Args:
            example: a dict[str, tf.Tensor], an individual rollout
        Returns:
            A dict[str, tf.Tensor].
        """
        return example

    # Public interface.

    def action_size(self):
        """Returns the dimensionality of the action space."""
        action_shape = self._environment().action_shape
        assert len(action_shape) == 1
        return action_shape[0]

    def parse_tfrecord(self, x, process_observations=True):
        """Parses a raw tfrecord byte string.

        We include the `process_observations` as rollouts can be quite long and
        observations can be quite large. If we want to only access a few observations
        from each rollout, then it is much cheaper to not process the observations
        when we read them from the tfrecord files but rather only the process the
        observations we have selected.

        Args:
            x: tf.Tensor with dtype tf.string, the tfrecord proto byte string
            process_observations: bool, whether or not to call `_process_observations`
                on the observations
        """
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
        """Returns a string glob pattern matching all tfrecord files in the split."""
        return self._tfrecords_pattern().format(split=split)

    def get_tfrecord_files(self, split):
        """Returns a string tf.Tensor with all tfrecord file names in the split."""
        return tf.io.matching_files(self.get_tfrecords_pattern(split=split))

    def rollouts_ds(self, split="train", process_observations=True):
        """Returns a tf.data.Dataset where each item is a full rollout.

        Each example is a dict containing tf.Tensor items with dtype, shape:
            'observations': _observation_dtype()?, [rollout_len, *_observation_shape()]
            'actions': tf.float32, [rollout_len, action_size()]
            'rewards': tf.float32, [rollout_len]

        Note that the `_process_observations` method can change the shape and
        dtype of the 'observations' item if the `process_observations` kwarg
        is set to true.

        Further note that the `_process_rollout` method has the potential to make
        arbitrary changes to each example.

        Please document these changes somewhere in the subclasses if they occur.

        Args:
            split: str, either "train" or "validation"
            process_observations: bool, please see documentation for the `parse_tfrecord`
                method for more information on this
        Returns:
            A tf.data.Dataset.
        """
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
        """Returns a dataset with random temporal slices of rollouts.

        The returned dataset will be similar to `rollouts_ds` except that every
        tensor will have a first dimension of size `slice_size`. This corresponds
        to a window of `slice_size` contiguous timesteps in the rollout. Please
        see the `rollouts_ds` documentation for more information about the returned
        dataset.

        The temporal windows are chosen at random positions. They are guaranteed to
        have a length of exactly `slice_size`.

        Args:
            slice_size: positive int, the length of the windows in time steps
            split: str, either "train" or "validation"
        Returns:
            A tf.data.Dataset.
        """

        def map_fn(x):
            x = processing.slice_example(x, slice_size=slice_size)
            x["observations"] = self._process_observations(x["observations"])
            return x

        ds = self.rollouts_ds(split=split, process_observations=False)
        return ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def random_rollout_observations(self, split="train", obs_sampled_per_rollout=100):
        """Returns a dataset containing single observations.

        Each example in the dataset is a dict with a single key "observation". The
        observation is chosen at random from a rollout.

        The purpose of the `obs_sampled_per_rollout` is to increase efficiency. Individual
        rollouts can be quite large, so training on a single observation per rollout
        can cause the data pipeline to become IO bound. By sampling multiple observations
        per rollout, we can speed up the pipeline while (hopefully) keeping the examples
        (mostly) IID.

        Args:
            split: str, either "train" or "validation"
            obs_sampled_per_rollout, bool
        Returns:
            A tf.data.Dataset.
        """

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
    """Abstract base class for rollout datasets coming from raw images.

    A raw image dataset is defined as one where the observations are stored
    as a byte tf.string tensor and have an actual dtype of tf.uint8. Observations
    in returned datasets will be of dtype float32 with values normalized in the
    range [0, 1].

    The main purpose of this class is convenience.
    """

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

    def _process_observations(self, observations):
        observations = tf.map_fn(
            functools.partial(tf.io.parse_tensor, out_type=tf.uint8),
            tf.squeeze(observations),
            dtype=tf.uint8,
        )
        # Convert to floats in the range [0, 1]
        observations = tf.cast(observations, tf.float32) / 255.0
        return observations


class EncodedRolloutDatasetBuilder(RolloutDatasetBuilder):
    """Abstract base class for rollout datasets with encoded observations.

    Such a dataset is defined as one where observations are tensors of rank 1
    and dtype float32. They are assumed to be stored as tf.float32 tensor and
    have tf.float32 as their true data type.

    Additional features, such as standard deviations of the encodings, can be included
    by overriding the `_additional_features` method. If doing this, please add
    documentation in the subclass.

    The main purpose of this class is convenience.
    """

    # Methods to potentially override.

    def _environment(self):
        raise NotImplementedError()

    def _tfrecords_pattern(self):
        raise NotImplementedError()

    def representation_size(self):
        """An int representing the dimensionality of the representation."""
        raise NotImplementedError()

    def _stored_observation_dtype(self):
        return tf.float32

    def _sample_observations(self, example):
        """Randomly sample observations for probabilistic representations.

        This method allows us to randomly sample observations if we so
        desire. For deterministic representations, it can just return the
        observation, as is the default here.

        Args:
            example: dict[str, tf.Tensor], a (slice of a) rollout
        Returns:
            A tf.Tensor representing a sampled observation.
        """
        return example["observations"]

    # Private methods.

    def _observation_shape(self):
        return (self.representation_size(),)

    # Public interface.

    def get_autoregressive_slices(
        self,
        sequence_length,
        sample_observations=False,
        difference_targets=False,
        split="train",
    ):
        """Returns a dataset for training autoregressive models.

        The goal of an autoregressive moded is to predict the next observation given
        a history of states, a history of actions, and the action taken at the current state.
        Essentially the task is to learn a mapping:
            (actions[:t], observations[:t]) --> observations[t]

        Args:
            sequence_length: a positive int, the length of the rollout slices
            sample_observations: bool, whether or not to sample from probabilistic representations.
                Has no effective for deterministic representations
            difference_targets: bool, whether to predict the difference between the next observation
                and the current observation or whether to predict the next observation directly
            split: str, either "train" or "validation"
        Returns:
            A tf.data.Dataset with 2-tuples as examples. The first item is concatenated observations
            and actions while the second item is the observations shifted in time by one.
        """
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
            if difference_targets:
                targets = o[1:] - o[:-1]
            else:
                targets = o[1:]
            inputs = tf.reshape(inputs, [sequence_length, rep_size + action_size])
            targets = tf.reshape(targets, [sequence_length, rep_size])
            return inputs, targets

        return self.random_rollout_slices_ds(
            slice_size=sequence_length + 1, split=split
        )
