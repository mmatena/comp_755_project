"""Common dataset processing functions."""
import tensorflow as tf
from rl755.data.common import augment


def standard_dataset_prep(ds, batch_size, repeat=True, shuffle_buffer_size=1000):
    """Common dataset preprocessing for training including batching and shuffling."""
    # TODO(mmatena): Add docs.
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def slice_example(x, slice_size):
    """Returns a random slice along the first dimension from tensors in example `x`.

    We assume that all values in `x` have an identical first dimension.

    Args:
        x: a dict of tf.Tensors
        slice_size: a positive integer
    Returns:
        A dict of of tf.Tensors.
    """
    # TODO(mmatena): Handle slice_sizes biggers than the rollout length.
    rollout_length = tf.shape(next(iter(x.values())))[0]
    rollout_length = tf.minimum(rollout_length, x["done_step"] + 1)
    slice_start = tf.random.uniform([], 0, rollout_length - slice_size, dtype=tf.int32)
    x = {
        k: v[slice_start : slice_start + slice_size]
        for k, v in x.items()
        if k != "done_step"
    }
    return x

def augment_for_train(image,
                         height,
                         width,
                         color_distort=True,
                         crop=False,
                         blur=True,
                         impl='simclrv2'):
    """Preprocesses the given image for training.
    Args:
        image: `Tensor` representing an image of arbitrary size.
        height: Height of output image.
        width: Width of output image.
        color_distort: Whether to apply the color distortion.
        crop: Whether to crop the image.
        flip: Whether or not to flip left and right of an image.
        impl: 'simclrv1' or 'simclrv2'.  Whether to use simclrv1 or simclrv2's
            version of random brightness.
    Returns:
        A preprocessed image `Tensor`.
    """
    if crop:
        # not implemented
        image = image
    if blur:
        image = augment.random_blur(image, height, width)
    if color_distort:
        image = augment.random_color_jitter(image, impl=impl)
    image = tf.reshape(image, [height, width, 3])
    image = tf.clip_by_value(image, 0., 1.)
    return image
