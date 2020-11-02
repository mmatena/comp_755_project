#!/bin/bash

#############################################################
# Various variables to set.
#############################################################
# TODO(mmatena): Make these flags.

# The directory of the cloned github repo.

MODEL_DIR=../tmp/bossfight_controller_model/
#############################################################

[ -d $MODEL_DIR ] || mkdir $MODEL_DIR



export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

python scripts/models/train_controller.py \
    --environment=bossfight \
    --model_dir=$MODEL_DIR \
    --memory_model=deterministic_lstm_64dm_32di\
    --vision_model=vae_32d \
    --rollout_max_steps=1000 \
    --cma_population_size=64 \
    --cma_trials_per_member=16 \
    --cma_steps=1000

