python scripts/data_gen/create_rollouts.py \
    --out_dir=../tmp/raw_bossfight/train \
    --num_rollouts=10000 \
    --environment=bigfish \
    --max_steps=1000 \
    --max_simul_envs=64
