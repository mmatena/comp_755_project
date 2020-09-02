#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
PROJECT_DIR=~/projects/comp_755_project

# MORE_FLAGS="--model='encoded_knn_rollout_transformer'\
#     --model_kwargs='{'k': 10, 'corpus_size': 1000, 'lambda_knn': 0.0}'"
#############################################################


unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0
module add gcc/9.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/longleaf/apps/gcc/9.1.0/lib64
# I copied the module to here since we can't access the original one in singularity.
LD_EXPORT_CMD="export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/package_hacks/gcc/9.1.0/lib64"

run_python() {
  echo "$LD_EXPORT_CMD && python $PROJECT_DIR/scripts/one_offs/transformer_eval.py \
      --model=encoded_knn_rollout_transformer \
      --model_kwargs={'k':10,'corpus_size':1000,'lambda_knn':0.0}"
}

run_singularity() {
  echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "\\\"$(run_python)\\\""
}

launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  # CMD=$(echo sbatch \
  #   --ntasks=1 \
  #   --time=0:30:00 \
  #   --mem=6g \
  #   --partition=volta-gpu \
  #   --gres=gpu:1 \
  #   --qos=gpu_access \
  #   --wrap="\"$(run_singularity)\"")
  CMD=$(echo sbatch \
    --ntasks=12 \
    --time=2:30:00 \
    --mem=12g \
    --partition=volta-gpu \
    --gres=gpu:1 \
    --qos=gpu_access \
    --wrap="\"$(run_singularity)\"")
  eval $CMD
}

# Run the command to actually launch the job.
launch
