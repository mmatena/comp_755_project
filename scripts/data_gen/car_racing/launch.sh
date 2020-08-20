#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# Where all of the rollouts and logs will be written to.
# Probably already needs to exist.
DATA_DIR=/pine/scr/m/m/mmatena/test_rollouts

NUM_ROLLOUTS=10
PARALLELISM=1
MAX_STEPS=2000

NUM_CORES=8
MEMORY=16g
#############################################################

# Parallelism 1: 
# Parallelism 2: 258.24171566963196 seconds
# Parallelism 5: 111.2401180267334 seconds


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
  # CMD=$(echo sbatch \
  #   --ntasks=1 \
  #   -c ${NUM_CORES} \
  #   --time=5:00:00 \
  #   --mem=${MEMORY} \
  #   --partition=general \
  #   --wrap="\"$(run_python)\"")
  CMD=$(echo sbatch \
    --ntasks=1 \
    -c ${NUM_CORES} \
    --error="$DATA_DIR/logs.err" \
    --output="$DATA_DIR/logs.out" \
    --time=5:00:00 \
    --mem=${MEMORY} \
    --partition=gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_python)\"")
  echo $CMD
  eval $CMD
}


launch
