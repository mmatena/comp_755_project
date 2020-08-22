#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

MODEL_DIR=/pine/scr/m/m/mmatena/test_vae_train

TRAIN_STEPS=2000

NUM_CORES=4
MEMORY=4g
TIME="8:30:00"
#############################################################


module add python/3.6.6

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/models/car_racing/train_vae.py \
    --model_dir=$MODEL_DIR \
    --train_steps=$TRAIN_STEPS
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=${NUM_CORES} \
    --error="$MODEL_DIR/logs-%j.err" \
    --output="$MODEL_DIR/logs-%j.out" \
    --time=${TIME} \
    --mem=${MEMORY} \
    --partition=gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_python)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch
