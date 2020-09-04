#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

MODEL_DIR=/pine/scr/m/m/mmatena/mog_ar_transformer_train

TRAIN_STEPS=250000

NUM_GPUS=1
NUM_CORES=16
MEMORY=16g
TIME="2-"
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/package_hacks/gcc/9.1.0/lib64
# I copied the module to here since we can't access the original one in singularity.
LD_EXPORT_CMD="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/package_hacks/gcc/9.1.0/lib64"

run_python() {
  echo "$LD_EXPORT_CMD && python $PROJECT_DIR/scripts/models/car_racing/train_ar_transformer.py \
      --model_dir=$MODEL_DIR \
      --train_steps=$TRAIN_STEPS \
      --batch_size=64"
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
    --gres=gpu:${NUM_GPUS} \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  echo $CMD
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
# Run the command to actually launch the job.
launch

