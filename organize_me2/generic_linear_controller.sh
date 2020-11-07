MODEL_DIR=../tmp/$env/controllers/linear_controller
#############################################################
[ -d ../tmp/$env/controllers/ ] || mkdir ../tmp/$env/controllers
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR



export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

python scripts/models/train_controller.py \
    --environment=$env \
    --model_dir=$MODEL_DIR \
    --memory_model=deterministic_transformer_32dm_32di \
    --vision_model=vae_32d \
    --rollout_max_steps=1000 \
    --cma_population_size=64 \
    --cma_trials_per_member=16 \
    --cma_steps=1000

