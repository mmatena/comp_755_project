

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
python scripts/data_gen/car_racing/parallel_rollout_main.py \
  --num_rollouts=12 \
  --parallelism=12 \
  --max_steps=1000 \
  --out_dir=tmp \
  --environment=TAKE_COVER \
  --policy=RandomIIDPolicy
