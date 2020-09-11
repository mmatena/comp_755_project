#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
PROJECT_DIR=~/projects/comp_755_project

NUM_CORES=4
MEMORY=4g
#############################################################

module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
# LD_EXPORT_CMD="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/package_hacks/gcc/9.1.0/lib64"

run_python() {
  # echo "python $PROJECT_DIR/scripts/util/gym_server.py"
  echo "python $PROJECT_DIR/scripts/util/parallel_gym_2.py"
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=${NUM_CORES} \
    --time=8:30:00 \
    --mem=${MEMORY} \
    --partition=volta-gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_python)\"")
  eval $CMD
}

# Run the command to actually launch the job.
launch
