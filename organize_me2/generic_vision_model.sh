
[ -d ../tmp/$env/vision_model ] || mkdir ../tmp/$env/vision_model

python scripts/models/train_vision_component.py \
   --model_dir="../tmp/$env/vision_model" \
   --train_steps=100000 \
   --environment=$env \
   --model=vae_32d
