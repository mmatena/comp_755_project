#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

MODEL_DIR=/pine/scr/m/m/mmatena/comp_755_project/data/bigfish/vae_32d_rollouts/train

TRAIN_STEPS=100000

NUM_CORES=8
MEMORY=8g
TIME="8:30:00"
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/data_gen/encode_raw_rollouts.py \
    --out_dir=$MODEL_DIR \
    --train_steps=$TRAIN_STEPS \
    --environment=bigfish \
    --vision_model=vae_32d \
    --estimated_num_rollouts=10000 \
    --split=train
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
    --partition=volta-gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch
