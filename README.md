# COMP 755 project

## Overview
This project was started by Hastings Greer and Michael Matena.
We plan to research model-based reinforcement learning (RL) and
create infrastructure to enable this.

### Research Goals
At this moment in time, our research goals haven't been fully specified, but our general
direction will be model-based RL.
Here are a few specific ideas that we have had:
- Using a GAN loss on the infilling u-net to capture multi-modality in the environment.
- Applying ideas from NLP to model-based RL:
    - Training a transformer on various autoregressive or infilling tasks.
    - Explicitly retrieving past experiences to get a better world model.

Collaborators are free to add their own ideas to our project.
We look forward to discussing research and exploring new ideas.

### Infrastructure Goals
We are using Python 3, TensorFlow 2, and OpenAI Gym.
When running on longleaf, please use the following modules
```bash
module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0
```

Some of our infrastructure goals include:
- Have an easy interface for launching and monitoring experiments.
- Make use of the parallelism and storage available on the longleaf cluster.
- Make switching between OpenAI Gym environments easy.

## Contributing
We use github issues for tracking experiments and feature implementations.
Our general workflow involves creating a tracking issue, opening a PR that references
the issue, and then merging that PR.
If you are modifying someone else's code, we recommend getting their review on the PR
before merging it, especially if much of the codebase depends on that code.

### Precommit
Please make use of the precommit.

### Code Organization
TODO

## Usage
TODO: Show examples of how to perform common actions such as defining new datasets, training a model,
evaluating a model, etc.

## (Some) dependencies

TODO: Move this info into a `requirements.txt`.

```bash
pip install pre-commit

pip install noise
pip install Box2D
pip install ray
pip install absl-py
pip install pyvirtualdisplay
pip install PyOpenGL PyOpenGL_accelerate

pip install tensorflow-probability==0.9.0
pip install cloudpickle==1.3.0

pip install cma

# For transformer
pip install bert-for-tf2
pip install mock
```