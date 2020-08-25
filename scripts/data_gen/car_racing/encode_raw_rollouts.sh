#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# 30 total shards  # Based on some rough calculations, this produces about 100MB per shard.
NUM_OUTER_SHARDS=4
NUM_SUBSHARDS=4
BASE_OUTER_SHARD_INDEX=0
# OUT_DIR=/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/encoded_rollouts
OUT_DIR=/pine/scr/m/m/mmatena/test_encoded_rollouts2
OUT_NAME=encoded_rollouts
MODEL="raw_rollout_vae_32ld"

NUM_GPUS=4
NUM_CORES=12
# Needs high memory due as full rollouts are large.
MEMORY=30g
TIME="8:30:00"
#############################################################


# volta-gpu:

# gpu:
# 1 GPU for 8 input shards: 80s
# 4 GPU for 8 input shards: 80s
# 8 GPU for 8 input shards: 83s sec

# 1 GPU for 64 input shards: 
# 8 GPU for 64 input shards: 



unset OMP_NUM_THREADS
SIMG_PATH=/nas/longleaf/apps/tensorflow_py3/2.1.0/simg
SIMG_NAME=tensorflow2.1.0-py3-cuda10.1-ubuntu18.04.simg

module add python/3.6.6
module add tensorflow_py3/2.1.0

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

run_python() {
  GPU_INDEX='$GPU_INDEX'
  echo "python $PROJECT_DIR/scripts/data_gen/car_racing/encode_raw_rollouts.py \
      --num_outer_shards=$NUM_OUTER_SHARDS \
      --outer_shard_index=$(($BASE_OUTER_SHARD_INDEX + ${GPU_INDEX})) \
      --num_sub_shards=$NUM_SUBSHARDS \
      --gpu_index=${GPU_INDEX} \
      --out_dir=$OUT_DIR \
      --out_name=$OUT_NAME \
      --model=$MODEL"
}


run_singularity() {
  echo singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME bash -c "\\\"$(run_python)\\\""
}

run_sh() {
  echo  "for ((GPU_INDEX=0;GPU_INDEX<=$NUM_GPUS;GPU_INDEX++)); \
            do $(run_singularity); \
          done"
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
    --wrap="\"$(run_sh)\"")
  echo $CMD
  eval $CMD
}


# Make the model directory if it does not exist.
[ -d $OUT_DIR ] || mkdir $OUT_DIR
# Run the command to actually launch the job.
launch

