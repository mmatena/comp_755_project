[ -d ../tmp/$env ] || mkdir ../tmp/$env
[ -d ../tmp/$env/raw_rollouts ] || mkdir ../tmp/$env/raw_rollouts

[ -d ../tmp/$env/raw_rollouts/test ] || mkdir ../tmp/$env/raw_rollouts/test
python scripts/data_gen/create_rollouts.py \
    --out_dir=../tmp/$env/raw_rollouts/test \
    --num_rollouts=1000 \
    --environment=$env \
    --max_steps=1000 \
    --max_simul_envs=64
