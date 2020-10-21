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


run_singularity() {
  echo singularity shell --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME 
}


launch() {
  # Not too sure why I have to do it like this, but just running the comman
  # causes it fail to launch.
  CMD=$(echo srun \
    --ntasks=10 \
    --time=1:00:00 \
    --mem=10g \
    --partition=gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --pty \
    $(run_singularity))
  eval $CMD
}

# Run the command to actually launch the job.
launch
