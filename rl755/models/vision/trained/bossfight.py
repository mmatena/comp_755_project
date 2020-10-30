
"""Trained vision components for the bossfight environment."""
from .. import instances
import getpass  

def vae_32d():
    # TODO: Add docs
    print("TODO: THIS MODEL ISN'T TRAINED. LOAD WEIGHTS!!!")
    model = instances.vae_32d()
    
    if getpass.getuser() == "tgreer":
        weights_path = "/playpen-raid1/tgreer/rl/tmp/bossfight_vision_model/model-050.hdf5"
    else:
        raise("Please add the weights path for your computer, then modify this function")

    model.load_weights(weights_path)
    return model
