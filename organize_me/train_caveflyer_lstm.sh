#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# MODEL=deterministic_lstm_32dm_32di
MODEL=deterministic_lstm_64dm_32di
MODEL_DIR=/pine/scr/m/m/mmatena/comp_755_project/models/memory/caveflyer/$MODEL

# Changing to 150k since it looked like the 32dm version didn't fully converge at 100k.
TRAIN_STEPS=150000

NUM_CORES=6
MEMORY=8g
TIME="22:30:00"
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/models/train_memory_component.py \
    --model_dir=$MODEL_DIR \
    --train_steps=$TRAIN_STEPS \
    --environment=caveflyer \
    --batch_size=256 \
    --model=$MODEL \
    --rollouts_dataset=EncodedRolloutsVae32d \
    --slices_per_rollout=2
}


run_singularity() {
  echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "\\\"$(run_python)\\\""
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
    --gres=gpu:4 \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch
