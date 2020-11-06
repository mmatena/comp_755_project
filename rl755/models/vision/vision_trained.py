from . import instances


def vae_32d(environment):
     # TODO: Add docs
     weights_path = (
         f"../tmp/{environment}/vision_model/model-100.hdf5"
     )
     model = instances.vae_32d()
     model.load_weights(weights_path)
     return model

print("in module", vae_32d)
