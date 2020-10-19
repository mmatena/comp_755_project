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

from rl755.models.car_racing import vae
from rl755.models.car_racing import transformer

import config


def raw_rollout_vae_32ld():
    """VAE trained on images from raw rollouts for 100k steps with
    a batch size of 256 for 100k steps and a latent dim of 32 and
    beta of 1.0. Used L2 reconstruction loss between pixels.

    Returns:
        A rl755.models.car_racing.vae.Vae instance.
    """
    weights_path = config.vae_weights_path
    model = vae.Vae(representation_size=32, beta=1.0)
    model.load_weights(weights_path)
    return model


def base_transformer_off_vae_32ld():
    # TODO: Add docs
    print("TODO: Update checkpoint once training is finished.")
    weights_path = "/pine/scr/m/m/mmatena/mse_ar_transformer_train/model-019.hdf5"
    input_size = 32 + 3  # latent_dim + action_dim
    sequence_length = 32  # Idk if it matters here.

    model = transformer.base_deterministic_transformer()
    # Build the model this way.
    model(tf.ones([1, sequence_length, input_size]))
    model.load_weights(weights_path)
    return model


# def encoded_knn_rollout_transformer(k, corpus_size, lambda_knn):
#     # TODO(mmatena): Add docs explaing all the parameters this was trained with.
#     weights_path = "/pine/scr/m/m/mmatena/mog_ar_transformer_train/model.hdf5"
#     seqlen = 32
#     input_size = 32 + 3  # latent_dim + action_dim
#     ar_transformer = encoded_rollout_transformer()
#     ar_transformer.return_layer_outputs = True
#     model = AutoregressiveLookupTransformer(
#         ar_transformer=ar_transformer,
#         knn_lookup=KnnLookup(k=k, num_points=corpus_size),
#         lambda_knn=lambda_knn,
#     )
#     model.build(input_shape=(None, seqlen, input_size))
#     model.load_ar_weights(weights_path)
#     return model