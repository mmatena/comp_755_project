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

run_shell() {
  echo sh -i
}

launch() {
  CMD=$(echo srun \
    --ntasks=8 \
    --time=2:30:00 \
    --mem=4g \
    --partition=general \
    --pty \
    $(run_shell))
  eval $CMD
}

# Run the command to actually launch the job.
launch
