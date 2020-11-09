""" Because the VAE encodes the information most common in the training images, the information in the residue is both rare, 
indicating importance to gameplay, and sparse, which allows threshhold then average-pool to capture it efficiently.
This even allows information never seen by the vision model to be exploited by the controller."""

from instances import vae
