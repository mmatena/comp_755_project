#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
PROJECT_DIR=~/projects/comp_755_project
#############################################################


unset OMP_NUM_THREADS
# SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
# SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.0.0/simg
SIMG_NAME=tensorflow2.0.0-py3-cuda10.0-ubuntu18.04.simg

module add python/3.6.6
# module add tensorflow_py3/2.1.0
module add tensorflow_py3/2.0.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/one_offs/transformer_eval.py
}

run_singularity() {
  echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "\\\"$(run_python)\\\""
}

launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=1 \
    --time=0:30:00 \
    --mem=6g \
    --partition=volta-gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}

# Run the command to actually launch the job.
launch
