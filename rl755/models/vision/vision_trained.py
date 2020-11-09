from . import instances


def vae_32d(environment):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/vae_32d/model-100.hdf5"
     )
     model = instances.vae_32d()
     model.load_weights(weights_path)
     return model

def vae_64d(environment):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/vae_64d/model-100.hdf5"
     )
     model = instances.vae_64d()
     model.load_weights(weights_path)
     return model
