import tensorflow as tf
from rl755.data.car_racing import encoded_rollouts
from rl755.models.car_racing import saved_models

for x in encoded_rollouts.get_rollouts_ds():
    break

a = x["actions"][:, :3]
o = x["observations"]
inputs = tf.concat([o, a], axis=-1)
inputs = tf.expand_dims(inputs, axis=0)

model = saved_models.encoded_rollout_transformer()
y = model(inputs)
var = tf.math.reduce_std(y, axis=-1)
var = tf.squeeze(var)
print(var[..., 3:])
