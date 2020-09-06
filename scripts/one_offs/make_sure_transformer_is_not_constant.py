import tensorflow as tf
from rl755.data.car_racing import encoded_rollouts
from rl755.models.car_racing import saved_models

for x in encoded_rollouts.get_rollouts_ds(split="validation"):
    break

a = x["actions"][:, :3]
o = x["observations"]
# inputs = tf.concat([o, a], axis=-1)
inputs = o
inputs = tf.expand_dims(inputs, axis=0)

model = saved_models.encoded_rollout_transformer()
y = model(inputs[:, 300:332])
var = tf.math.reduce_std(y, axis=-2)
var = tf.squeeze(var)
print(var)
# targets = o[1:] - o[:-1]

mix = model.get_mix_of_gauss(y)
# print(-mix.log_prob([targets[200:232]]))
# print(-mix.log_prob([targets[20:52]]))
print(-mix.log_prob(inputs[:, 200:232]))
print(-mix.log_prob(inputs[:, 20:52]))

# var = tf.math.reduce_std(targets[200:232], axis=-2)
# var = tf.squeeze(var)
# print(var)
