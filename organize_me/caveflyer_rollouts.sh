#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

MODEL_DIR=/pine/scr/m/m/mmatena/comp_755_project/data/caveflyer/raw_rollouts/train

#############################################################


module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR


run_python() {
  echo python $PROJECT_DIR/scripts/data_gen/create_rollouts.py \
    --out_dir=$MODEL_DIR \
    --num_rollouts=10000 \
    --environment=caveflyer \
    --max_steps=1000
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --error="$MODEL_DIR/logs-%j.err" \
    --output="$MODEL_DIR/logs-%j.out" \
    --ntasks=8 \
    --time=2:30:00 \
    --mem=8g \
    --partition=general \
    --wrap="\"$(run_python)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch
