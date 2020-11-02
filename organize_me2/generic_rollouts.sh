[ -d ../tmp/$env ] || mkdir ../tmp/$env

[ -d ../tmp/$env/train ] || mkdir ../tmp/$env/train
python scripts/data_gen/create_rollouts.py \
    --out_dir=../tmp/$env/train \
    --num_rollouts=10000 \
    --environment=$env \
    --max_steps=1000 \
    --max_simul_envs=64
