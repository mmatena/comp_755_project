#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
PROJECT_DIR=~/projects/comp_755_project
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/longleaf/apps/gcc/9.1.0/lib64
# I copied the module to here since we can't access the original one in singularity.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/package_hacks/gcc/9.1.0/lib64

# run_singularity() {
#   echo singularity shell --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME 
# }
run_singularity() {
  echo sh -i
}

launch() {
  # Not too sure why I have to do it like this, but just running the comman
  # causes it fail to launch.
  CMD=$(echo srun \
    --ntasks=12 \
    --time=2:30:00 \
    --mem=20g \
    --partition=volta-gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --pty \
    $(run_singularity))
  # CMD=$(echo srun \
  #   --ntasks=12 \
  #   --time=2:30:00 \
  #   --mem=16g \
  #   --partition=general \
  #   --pty \
  #   $(run_singularity))
  # CMD=$(echo srun \
  #   --ntasks=4 \
  #   --time=2:30:00 \
  #   --mem=4g \
  #   --partition=general \
  #   --pty \
  #   $(run_singularity))
  eval $CMD
}
# 1: 0.13
# 2: 0.064


# Run the command to actually launch the job.
launch
