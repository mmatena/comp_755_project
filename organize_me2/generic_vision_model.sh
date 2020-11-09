
[ -d ../tmp/$env/vision_model ] || mkdir ../tmp/$env/vision_model

model_name=vae_32d

export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR

[ -d ../tmp/$env/vision_model/$model_name ] || mkdir ../tmp/$env/vision_model/$model_name

python scripts/models/train_vision_component.py \
   --model_dir="../tmp/$env/vision_model/$model_name" \
   --train_steps=100000 \
   --environment=$env \
   --model=$model_name
