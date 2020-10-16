
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR
python scripts/data_gen/car_racing/rollout_pickles_to_tfrecords.py \
	--parallelism=12 
	
