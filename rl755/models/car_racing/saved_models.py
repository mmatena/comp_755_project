"""Utility/book keeping to access models we've trained.

The reason this class exists is because I'm not sure of the best
practices for doing this yet.

We save the weights of the models only during training and store the
mapping between parameter files and model architectures here.

Doing it this way let's us get the tf.kera.models.Models subclass
object at the cost of having to keep track of the mapping between
saved parameters and architecture manually.
"""
from bert.transformer import TransformerEncoderLayer
import tensorflow as tf

from rl755.models.car_racing.vae import Vae
from rl755.models.common.transformer import AutoregressiveTransformer


def raw_rollout_vae_32ld():
    """VAE trained on images from raw rollouts for 100k steps with
    a batch size of 256 for 100k steps and a latent dim of 32 and
    beta of 1.0. Used L2 reconstruction loss between pixels.

    Returns:
        A rl755.models.car_racing.vae.Vae instance.
    """
    weights_path = "/pine/scr/m/m/mmatena/test_vae_train/model.hdf5"
    vae = Vae(latent_dim=32, beta=1.0)
    vae.load_weights(weights_path)
    return vae


def encoded_rollout_transformer():
    # TODO(mmatena): Add docs explaing all the parameters this was trained with.
    weights_path = "/pine/scr/m/m/mmatena/test_ar_transformer_train/model.hdf5"
    seqlen = 32
    input_size = 32 + 4 + 1  # latent_dim + action_dim + reward_dim
    output_size = 32 + 1  # latent_dim + reward_dim
    num_attention_heads = 12
    hidden_size = 768
    transformer_params = TransformerEncoderLayer.Params(
        num_layers=12,
        hidden_size=hidden_size,
        hidden_dropout=0.1,
        intermediate_size=4 * hidden_size,
        intermediate_activation="gelu",
        num_heads=num_attention_heads,
        size_per_head=int(hidden_size / num_attention_heads),
    )
    model = AutoregressiveTransformer(transformer_params, output_size=output_size)
    model.build([None, seqlen, input_size])
    model.load_weights(weights_path)
    return model
