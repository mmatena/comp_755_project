
[ -d ../tmp/$env/vision_model ] || mkdir ../tmp/$env/vision_model


export PYTHONPATH=$PYTHONPATH:$(pwd)

[ -d ../tmp/$env/vision_model/$model_name ] || mkdir ../tmp/$env/vision_model/$model_name

python scripts/models/train_vision_component.py \
   --model_dir="../tmp/$env/vision_model/$model_name" \
   --train_steps=100000 \
   --environment=$env \
   --model=$model_name
