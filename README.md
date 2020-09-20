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

## Running experiments
Our general experimental flow is based on the [World Models](https://arxiv.org/pdf/1803.10122.pdf) paper.
Here is an overview:
1. Collect a large number of rollouts in an environment via a random policy.
2. Learn a compact representation of individual observations from the rollouts by, for example, training a VAE.
3. Create a new dataset from the rollouts with observations replaced by their representation.
    This is for computational and memory savings.
4. Learn a sequential model of the dynamics of the encoded representations by, for example, training an autoregressive transformer.
5. Define a controller architecture that takes in representations from both the representation model
    and the sequential model and outputs an action to maximize rewards through, for example, CMA-ES.

In the following subsections I'll go over instructions on how to perform each step.

### Collecting random policy rollouts

#### Generating pickled rollouts
To generate rollouts with a random policy, use
the [`scripts/scripts/data_gen/car_racing/parallel_rollout_main.py`](https://github.com/mmatena/comp_755_project/blob/master/scripts/data_gen/car_racing/parallel_rollout_main.py) script.
This script will run rollouts in parallel and save them as pickled `Rollout` objects in a specified directory.
It is safe to have several instances of this script running in parallel with the same output directory.

Each pickle file will contain a list with a single `Rollout` in it.
Right now individual rollouts are very big, so we are fine saving them one-by-one in separate files.
Saving a list will allow us to put multiple rollouts in a single pickle file if rollouts are small for some
future task.

The script takes the following flags:
- `--num_rollouts` The number of rollouts that the script will generate.
- `--parallelism` A positive integer specifying the number of rollouts to do in parallel. It's probably best to keep this number a little under your number of cores.
- `--max_steps` The maximum number of steps to simulate each rollout for. Generated rollouts can be shorter than this if the task finishes early.
- `--out_dir` The directory to write the pickled rollouts to. Each rollout will be in a separate pickle file with a randomly generated name and a `.pickle` extension.
- `--environment` The name of the [`Environment`](https://github.com/mmatena/comp_755_project/blob/master/rl755/environments.py) enum to do rollouts on. For example, `CAR_RACING`. *Note: I've only tried this yet with car racing.*
- `--policy` The name of the policy to use for these rollouts. The policy will be accessed in a manner equivalent to `policy = rl55.models.$environment.policies.$policy()`. It should subclass [`rl755.data_gen.gym_rollouts.Policy`](https://github.com/mmatena/comp_755_project/blob/master/rl755/data_gen/gym_rollouts.py) or be a function returning an instance of such a class. In either case, it will be called with no arguments.
*Note: I've only tested this with a simple random policy. This might be inefficient when the policy involves expensive operations.*

The World Models papers generates 10000 random rollouts.
In order to do this efficiently, parallelism is crucial.
If you are having trouble scheduling large jobs, it is safe to run multiple instances of this script with the same output directory (and `num_rollouts` evenly divided between them) in different slurm jobs to get further parallelism.
I think I was able to get 10000 rollouts for car racing in around 8-10 hours by doing this.

**Note: I've been having trouble getting `ray` working recently. If you are having trouble, please reach out to mmatena to see if there is any solution yet.**

#### Converting pickles to tf records
The previous script will generate a directory full of pickle files.
We now want to convert those pickle files to a sharded `.tfrecord` file so that we can
efficiently access them in a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

To do this, we use the [`scripts/data_gen/car_racing/rollout_pickles_to_tfrecords.py`](https://github.com/mmatena/comp_755_project/blob/master/scripts/data_gen/car_racing/rollout_pickles_to_tfrecords.py) script.
It takes all of the `.pickle` files in a directory (which are all assumed to contain `Rollout`s) and converts them into a sharded `.tfrecord` file in another directory.

The script takes the following flags:
- `--parallelism` A positive integer specifying the number of processes to use. The conversion task is embarrassingly parallel, so you'll get a near linear speed up from increasing this.
- `--pickles_per_tfrecord_file` The number of rollouts to put in each shard of the `.tfrecord` file. We want shards to be roughly 100-200 MB. Missing this exact range isn't a big deal. The best way to estimate a good value for this flag is to copy a single pickled rollout to another directory and run this script on it. By looking at the size of the generated `.tfrecord` file, you'll have an idea of how many rollouts equals 100-200 MB.
- `--pickle_dir` The directory containing the pickled rollouts. Every `.pickle` file in this directory is assumed to contain a Python list with a single `Rollout` in it. Failure to meet this condition will probably result in the script crashing.
- `--out_dir` The directory to write the sharded `.tfrecord` file to.
- `--out_name` Name given to the generated `.tfrecord` files. A complete file name will look something like `$out_name.tfrecord-00003-of-00074`.

#### Removing corrupted tf records
For some reason, the previous script sometimes generates corrupted `.tfrecord` shards.

The way I've dealt with this is by running the [`scripts/data_gen/find_corrupted_tfrecords.py`](https://github.com/mmatena/comp_755_project/blob/master/scripts/data_gen/find_corrupted_tfrecords.py) script.
It takes a single flag `--pattern`, which is the glob pattern of files to check.

Set it to a value that will match all of the generated files in the previous step, for example `$out_dir/$out_name.tfrecord*`.
It will print the names of files with corrupted data.

I then just delete those files.
Having 10000 rollouts is a bit overkill, so we can handle losing a little data.

If most of your files are corrupted though, then something probably went wrong and you'll want to look into it.

#### Adding access to the rollouts in the code
Once you have generated the dataset, we now add a way to access it as a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

You should create `rl755/data/$environment/raw_rollouts.py` file.
See [`rl755/data/car_racing/raw_rollouts.py`](https://github.com/mmatena/comp_755_project/blob/master/rl755/data/car_racing/raw_rollouts.py) for an example.
The next step, learning representations of observations, will require a function named `random_rollout_observations` within this file.
Please see the next section for more details.

To do this for a different environment, there are only a few things you might need to change.
It might make sense just to put the common code in a shared directory sometime in the future.
If you get to this point, feel free to reach out to mmatena for help.


### Learning representations of observations
The goal of this step is to learn a function that takes raw observations
and maps them to representations with "nice" structure and a much lower dimensionality.

You can use the [`scripts/models/train_obs_encoder.py`](https://github.com/mmatena/comp_755_project/blob/master/scripts/models/train_obs_encoder.py) script to learn such a model.
You should train your model on the `volta-gpu` or `gpu` longleaf partitions.
This script currently does not have support for multi-GPU training.

There are a couple of requirements in order to use this script:
- You must define a model class that extends the [`ObservationEncoder`](https://github.com/mmatena/comp_755_project/blob/master/rl755/models/encoder/common.py) class and implements its abstract methods. Please see the class definition for more details.
- You'll need to have a file `rl755/data/$environment/raw_rollouts.py` with a method `random_rollout_observations` that takes a kwarg `obs_sampled_per_rollout`. The dataset should be comprised of individual raw observations from the collected rollouts.
The `obs_sampled_per_rollout` argument is explained later in this section.

The script takes the following flags:
- `--model_dir` The directory where we save model checkpoints and tensorboard logs.
- `--train_steps` The number of batches to train for.
- `--environment` The name of the [`Environment`](https://github.com/mmatena/comp_755_project/blob/master/rl755/environments.py) enum from which the rollouts come from. For example, `CAR_RACING`.
- `--model` The name of the model to train. The model will be accessed in a manner equivalent to `model = rl55.models.$environment.policies.$model(obs_sampled_per_rollout=<>)`. Note that this flag can contain periods to allow access to more deeply nested classes/functions. For example, `vae.Vae` would be valid. The model must subclass [`ObservationEncoder`](https://github.com/mmatena/comp_755_project/blob/master/rl755/models/encoder/common.py) or be a function returning an instance of such a class.
- `--batch_size` The number of raw observations in each mini batch.
- `--representation_size` A positive integer representing the dimensionality of encoded representation. It will be passed as a kwarg to the model.
- `--save_every_n_steps` Save a checkpoint every this number of training steps. Note that we are saving the weights only. The checkpoint file names will be formatted as `model-{checkpoint_number:03d}.hdf5`. The train loss will also only be logged to tensorboard every this number of training steps due to how keras works.
- `--obs_sampled_per_rollout` A positive integer. The model will take this many observations at random each time it encounters a rollout. It will be passed as a kwarg to the `random_rollout_observations` function. We include this as individual rollouts can be very big. Ideally we'd only sample a single observation from each rollout to prevent potential correlations in a minibatch, but the large size of rollouts leads to the input becoming a bottleneck. Sampling multiple observations from each rollout lets us do more computation per disc read operation, making training much faster. As for the value to use, I've been using 100 for car racing, which tends to have rollouts of length about 1000. I haven't really experimented with this, but it seems like it's worked OK. I'd recommend keeping it signficantly smaller than the shuffle buffer size, which is 1000 in this script.
- `--learning_rate` A positive float. The learning rate to use during training.

**Note: I actually haven't used this script before. I used the [`scripts/models/car_racing/train_vae.py`](https://github.com/mmatena/comp_755_project/blob/master/scripts/models/car_racing/train_vae.py) script to train a VAE for car racing. I'll try actually running this soon.**

### Create a dataset of rollouts with encoded observations
TODO


## Existing models and datasets
TODO, add something about configs


## General tips
TODO, put the srun for both GPU and regular, the `-i` and `-m pdb`, dealing with `NaN`s during training, my transformer issues,
my slurm stuff



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
pip install rpyc

# For transformer
pip install bert-for-tf2
pip install mock
```