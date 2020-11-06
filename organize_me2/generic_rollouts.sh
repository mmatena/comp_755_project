[ -d ../tmp/$env ] || mkdir ../tmp/$env
[ -d ../tmp/$env/raw_rollouts ] || mkdir ../tmp/$env/raw_rollouts

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
[ -d ../tmp/$env/raw_rollouts/train ] || mkdir ../tmp/$env/raw_rollouts/train
python scripts/data_gen/create_rollouts.py \
    --out_dir=../tmp/$env/raw_rollouts/train \
    --num_rollouts=5000 \
    --environment=$env \
    --max_steps=1000 \
    --max_simul_envs=64
