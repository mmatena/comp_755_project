#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# MODEL_DIR=/pine/scr/m/m/mmatena/tmp/caveflyer_controller_test
# MODEL_DIR=/pine/scr/m/m/mmatena/tmp/caveflyer_controller_det_test2
MODEL_DIR=/pine/scr/m/m/mmatena/tmp/caveflyer_controller_64_det_test2

NUM_CORES=16
MEMORY=16g
TIME="1-"
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/models/train_controller.py \
    --environment=caveflyer \
    --model_dir=$MODEL_DIR \
    --memory_model=deterministic_transformer_64dm_32di \
    --vision_model=vae_32d \
    --rollout_max_steps=1000 \
    --cma_population_size=64 \
    --cma_trials_per_member=16 \
    --max_simul_envs=128 \
    --cma_steps=1000
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
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch
