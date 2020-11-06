[ -d ../tmp/$env ] || mkdir ../tmp/$env
[ -d ../tmp/$env/enc_rollouts ] || mkdir ../tmp/$env/enc_rollouts

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
[ -d ../tmp/$env/enc_rollouts/train ] || mkdir ../tmp/$env/enc_rollouts/train

python scripts/data_gen/encode_raw_rollouts.py \
    --out_dir=../tmp/$env/enc_rollouts/train \
    --environment=$env \
    --vision_model=vae_32d \
    --estimated_num_rollouts=10000 \
    --split=train
