#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

NUM_SHARDS=1
# OUT_DIR=/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/encoded_rollouts
OUT_DIR=/pine/scr/m/m/mmatena/test_encoded_rollouts
OUT_NAME=encoded_rollouts
MODEL="raw_rollout_vae_32ld"
BATCH_SIZE=2

NUM_GPUS=1
NUM_CORES=12
# Needs high memory due as full rollouts are large.
MEMORY=32g
TIME="8:30:00"
#############################################################



unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  echo python $PROJECT_DIR/scripts/data_gen/car_racing/encode_raw_rollouts.py \
    --num_shards=$NUM_SHARDS \
    --out_dir=$OUT_DIR \
    --out_name=$OUT_NAME \
    --model=$MODEL \
    --batch_size=$BATCH_SIZE
}


run_singularity() {
  echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "\\\"$(run_python)\\\""
}

launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=${NUM_CORES} \
    --error="$OUT_DIR/logs-%j.err" \
    --output="$OUT_DIR/logs-%j.out" \
    --time=${TIME} \
    --mem=${MEMORY} \
    --partition=volta-gpu \
    --gres=gpu:${NUM_GPUS} \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $OUT_DIR ] || mkdir $OUT_DIR
# Run the command to actually launch the job.
launch

