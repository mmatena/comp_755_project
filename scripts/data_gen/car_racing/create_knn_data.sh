#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# Let's go for 80 shards total.
NUM_OUTER_SHARDS=8
NUM_SUBSHARDS=10
BASE_OUTER_SHARD_INDEX=$1

OUT_DIR=/pine/scr/m/m/mmatena/comp_755_project/test_knn_data
OUT_NAME=knn_data

NUM_PASSES=100

NUM_GPUS=4
NUM_CORES=8
MEMORY=16g
TIME="23:30:00"
#############################################################

unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python_single() {
  GPU_INDEX=$1
  echo "python $PROJECT_DIR/scripts/data_gen/car_racing/create_knn_data.py \
      --num_outer_shards=$NUM_OUTER_SHARDS \
      --outer_shard_index=$(($BASE_OUTER_SHARD_INDEX + $GPU_INDEX)) \
      --num_sub_shards=$NUM_SUBSHARDS \
      --gpu_index=$GPU_INDEX \
      --out_dir=$OUT_DIR \
      --out_name=$OUT_NAME \
      --num_passes=$NUM_PASSES"
}

run_python() {
  echo "echo ''"
    for ((i=0;i<$NUM_GPUS;i++));
      do echo " & $(run_python_single $i) ";
    done
    echo " && fg"
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
  echo $CMD
  eval $CMD
}

# Make the model directory if it does not exist.
[ -d $OUT_DIR ] || mkdir $OUT_DIR
# Run the command to actually launch the job.
launch

