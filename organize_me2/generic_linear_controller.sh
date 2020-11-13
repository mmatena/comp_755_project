MODEL_DIR=../tmp/$env/controllers/linear_controller3
#############################################################
[ -d $MODEL_DIR ] || mkdir -p $MODEL_DIR



export PYTHONPATH=$PYTHONPATH:$(pwd)

python scripts/models/train_controller.py \
    --environment=$env \
    --model_dir=$MODEL_DIR \
    --memory_model=deterministic_transformer_64dm_64di \
    --vision_model=residual_vae_64d \
    --rollout_max_steps=1000 \
    --cma_population_size=64 \
    --cma_trials_per_member=16 \
    --cma_steps=1000

