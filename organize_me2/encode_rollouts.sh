[ -d ../tmp/$env ] || mkdir ../tmp/$env
[ -d ../tmp/$env/res_enc_rollouts ] || mkdir ../tmp/$env/res_enc_rollouts

export PYTHONPATH=$PYTHONPATH:$(pwd)
[ -d ../tmp/$env/res_enc_rollouts/train ] || mkdir ../tmp/$env/res_enc_rollouts/train

python scripts/data_gen/encode_raw_rollouts.py \
    --out_dir=../tmp/$env/res_enc_rollouts/train \
    --environment=$env \
    --vision_model=residual_vae_64d \
    --estimated_num_rollouts=10000 \
    --split=train
