#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# Where all of the rollouts and logs will be written to.
# Probably already needs to exist.
DATA_DIR=/pine/scr/m/m/mmatena/1000_rollouts_test

NUM_ROLLOUTS=500
PARALLELISM=10
MAX_STEPS=2000

NUM_CORES=12
MEMORY=10g
#############################################################


module add python/3.6.6

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/data_gen/car_racing/parallel_rollout_main.py \
    --num_rollouts=$NUM_ROLLOUTS \
    --parallelism=$PARALLELISM \
    --max_steps=$MAX_STEPS \
    --out_dir=$DATA_DIR
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=${NUM_CORES} \
    --error="$DATA_DIR/logs-%j.err" \
    --output="$DATA_DIR/logs-%j.out" \
    --time=4:00:00 \
    --mem=${MEMORY} \
    --partition=general \
    --wrap="\"$(run_python)\"")
  # CMD=$(echo sbatch \
  #   --ntasks=${NUM_CORES} \
  #   --error="$DATA_DIR/logs-%j.err" \
  #   --output="$DATA_DIR/logs-%j.out" \
  #   --time=4:00:00 \
  #   --mem=${MEMORY} \
  #   --partition=gpu \
  #   --gres=gpu:1 \
  #   --qos=gpu_access \
  #   --wrap="\"$(run_python)\"")
  eval $CMD
}


# Run the command to actually launch the job.
launch
