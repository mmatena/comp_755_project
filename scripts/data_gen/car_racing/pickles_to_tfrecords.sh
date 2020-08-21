#!/bin/bash

# NOTE: This tends to run pretty fast.

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.
PROJECT_DIR=~/projects/comp_755_project

# Where the pickle files reside. All pickle files in this directory are
# assumed to be lists of Rollout instances and will be put into tfrecords.
PICKLE_DIR=/pine/scr/m/m/mmatena/1000_rollouts_test

# Where we will write the tfrecords and logs to.
OUT_DIR=/pine/scr/m/m/mmatena/comp_755_project/data/car_racing/raw_rollouts

# Tfrecord files will be named like "$OUTNAME.tfrecord-*-of-*".
OUT_NAME=raw_rollouts

# Number of pickled files to put in each tfrecord file. We want shards to be
# around 100-200 MB. Setting this to 6 seems good for raw rollouts.
PICKLES_PER_TF_RECORD_FILE=6

PARALLELISM=10

NUM_CORES=12
MEMORY=32g
#############################################################


module add python/3.6.6

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR


run_python() {
  echo python $PROJECT_DIR/scripts/data_gen/car_racing/rollout_pickles_to_tfrecords.py \
    --parallelism=$PARALLELISM \
    --pickles_per_tfrecord_file=$PICKLES_PER_TF_RECORD_FILE \
    --pickle_dir=$PICKLE_DIR \
    --out_dir=$OUT_DIR \
    --out_name=$OUT_NAME
}


launch() {
  # Not too sure why I have to do it like this, but just running the command
  # causes it fail to launch.
  CMD=$(echo sbatch \
    --ntasks=${NUM_CORES} \
    --error="$OUT_DIR/logs-%j.err" \
    --output="$OUT_DIR/logs-%j.out" \
    --time=2:30:00 \
    --mem=${MEMORY} \
    --partition=general \
    --wrap="\"$(run_python)\"")
  eval $CMD
}


# Run the command to actually launch the job.
launch
