import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def _get_encoder(latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=6, strides=(3, 3), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Conv2D(
                filters=256, kernel_size=4, strides=(2, 2), activation="relu"
            ),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(2 * latent_dim),
        ]
    )


def _get_decoder(latent_dim):
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            # No activation
            tf.keras.layers.Dense(units=1024),
            tf.keras.layers.Reshape(target_shape=(1, 1, 1024)),
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=5, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=5, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=6, strides=2, activation="relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=9, strides=3, activation="sigmoid"
            ),
        ]
    )


class Vae(tf.keras.Model):
    def __init__(
        self,
        latent_dim=32,
        beta=1.0,
        log_losses=False,
        start_step=0,
        name="vae",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.log_losses = log_losses
        self.encoder = _get_encoder(latent_dim)
        self.decoder = _get_decoder(latent_dim)
        self.prior = tfd.MultivariateNormalDiag(
            tf.zeros(latent_dim), tf.ones(latent_dim)
        )
        self.step = tf.Variable(start_step, trainable=False, dtype=tf.int64)

    def x_from_data(self, data):
        return data["observation"]

    def encode(self, x):
        mean, prevar = tf.split(self.encoder(x), num_or_size_splits=2, axis=-1)
        return tfd.MultivariateNormalDiag(loc=mean, scale_diag=tf.nn.softplus(prevar))

    def decode(self, z):
        logits = self.decoder(z)
        return logits

    def sample_unconditionally(self, num_samples=1):
        z = self.prior.sample(num_samples)
        return self.decode(z)

    def get_losses(self, x):
        posterior = self.encode(x)
        reconstruction = self.decode(posterior.sample())

        # TODO(mmatena): Figure out the right combination of reduce_mean and
        # reduce_sum to use here.
        loss_recon = tf.reduce_sum(tf.square(x - reconstruction))
        loss_recon /= tf.cast(tf.shape(x)[0], tf.float32)
        loss_kl = tf.reduce_mean(tfd.kl_divergence(posterior, self.prior))
        return loss_recon, loss_kl

    def train_step(self, data):
        self.step.assign_add(1)

        x = self.x_from_data(data)
        with tf.GradientTape() as tape:
            loss_recon, loss_kl = self.get_losses(x)

            loss = loss_recon + self.beta * loss_kl

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        if self.log_losses:
            tf.summary.scalar("loss", data=loss, step=self.step)
            tf.summary.scalar("l2_loss", data=loss_recon, step=self.step)
            tf.summary.scalar("kl_loss", data=loss_kl, step=self.step)
        return {"loss": loss, "l2": loss_recon, "kl": loss_kl}
