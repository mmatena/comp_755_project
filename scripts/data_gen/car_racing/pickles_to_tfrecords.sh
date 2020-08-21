#!/bin/bash

# #############################################################
# # Various variables to set.
# #############################################################
# # TODO(mmatena): Make these flags.

# # The directory of the cloned github repo.
# PROJECT_DIR=~/projects/comp_755_project

# # Where all of the rollouts and logs will be written to.
# # Probably already needs to exist.
# DATA_DIR=/pine/scr/m/m/mmatena/1000_rollouts_test

# # NUM_ROLLOUTS=500
# # PARALLELISM=10
# # MAX_STEPS=2000

# # NUM_CORES=12
# # MEMORY=10g

# NUM_ROLLOUTS=1500
# PARALLELISM=10
# MAX_STEPS=2000

# NUM_CORES=12
# MEMORY=8g
# #############################################################


PROJECT_DIR=~/projects/comp_755_project
python $PROJECT_DIR/scripts/data_gen/car_racing/rollout_pickles_to_tfrecords.py \
  --parallelism=1 \
  --pickles_per_tfrecord_file=8 \
  --pickle_dir=/pine/scr/m/m/mmatena/pickled_rollouts_to_tfrecord_test \
  --out_dir=/pine/scr/m/m/mmatena/pickled_rollouts_to_tfrecord_test \
  --out_name=raw_rollouts_test

