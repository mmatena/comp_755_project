
MODEL_DIR=../tmp/$env/memory/deterministic_transformer_32dm_32di
[ -d ../tmp/$env/memory ] || mkdir ../tmp/$env/memory
[ -d $MODEL_DIR ] || mkdir $MODEL_DIR
TRAIN_STEPS=100000

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
python scripts/models/train_memory_component.py \
    --model_dir=$MODEL_DIR \
    --train_steps=$TRAIN_STEPS \
    --environment=$env \
    --batch_size=256 \
    --model=deterministic_transformer_32dm_32di \
    --rollouts_dataset=EncodedRolloutsVae32d \
    --slices_per_rollout=2
# Make the model directory if it does not exist.
# Run the command to actually launch the job.
