"""Created purely because I couldn't figure out how to get
this work on longleaf's notebook."""
import itertools
from pydoc import locate

from absl import app
from absl import flags
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "environment",
    None,
    "Which environment we are using rollouts from.",
)
flags.DEFINE_string(
    "model",
    None,
    "",
)

flags.DEFINE_integer(
    "num_samples", 1, "The number of times to sample from the VAE.", lower_bound=1
)
flags.DEFINE_bool("unconditional", False, "Whether to condition on real images or not.")

flags.mark_flag_as_required("environment")
flags.mark_flag_as_required("model")


def float_to_uint(x):
    return tf.cast(255.0 * x, tf.uint8)


def get_model():
    model = locate(f"rl755.models.vision.trained.{FLAGS.environment}.{FLAGS.model}")
    return model()


def get_dataset():
    dsb_cls = locate(f"rl755.data.envs.{FLAGS.environment}.RawRollouts")
    return dsb_cls().random_rollout_observations()


def main(_):
    vae = get_model()
    if FLAGS.unconditional:
        images = vae.sample_unconditionally(FLAGS.num_samples)
    else:
        inputs = [
            x["observation"] for x in itertools.islice(get_dataset(), FLAGS.num_samples)
        ]
        inputs = tf.stack(inputs, axis=0)
        # TODO(mmatena): Compare .sample() vs .mean().
        z = vae.encode(inputs).mean()
        images = vae.decode(z)

        # print(tf.io.serialize_tensor(float_to_uint(inputs)).numpy())
        images = float_to_uint(inputs).numpy()
        image = Image.fromarray(images[0])
        image.show()
        for _ in range(10):
            print("_")

    images = float_to_uint(images)
    print(tf.io.serialize_tensor(images).numpy())


if __name__ == "__main__":
    app.run(main)
