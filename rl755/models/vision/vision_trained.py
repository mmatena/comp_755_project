from . import instances


def vae_32d(environment, **kwargs):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/vae_32d/model-100.hdf5"
     )
     model = instances.vae_32d(environment, **kwargs)
     model.load_weights(weights_path)
     return model

def vae_64d(environment, **kwargs):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/vae_64d/model-100.hdf5"
     )
     model = instances.vae_64d(environment, **kwargs)
     model.load_weights(weights_path)
     return model
     
def residual_vae_64d(environment, **kwargs):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/residual_vae_64d/model-100.hdf5"
     )
     model = instances.residual_vae_64d(environment, **kwargs)
     model.load_weights(weights_path)
     return model
