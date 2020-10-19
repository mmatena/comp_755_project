import time

from absl import app
from absl import flags

import gym
import gym3

from procgen import ProcgenGym3Env

FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     "env_wertwret", "coinrunner", ""
# )

# flags.DEFINE_integer(
#     "steps", 100, ""
# )
# flags.DEFINE_string(
#     "env", "coinrunner", ""
# )

# run_gym3("coinrun", steps=1000, num=128)
# 4 cpu: 6.3 s
# 8 cpu: 5.9 s


def run_gym(env_name, steps=100):
    env_name = f"procgen:procgen-{env_name}-v0"
    start = time.time()
    env = gym.make(env_name, start_level=0, num_levels=1)
    observation = env.reset()
    for _ in range(steps):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
    del observation
    print(f"{time.time() - start} s")


def run_gym3(env_name, num=1, steps=100):
    start = time.time()
    env = ProcgenGym3Env(num=num, env_name=env_name, start_level=0, num_levels=1)
    for _ in range(steps):
        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        rew, obs, first = env.observe()
    print(f"{time.time() - start} s")


def main(_):
    env_name = "coinrun"
    start = time.time()
    # run_gym(env_name=env_name, steps=FLAGS.steps)
    run_gym(env_name=env_name)
    print(f"{time.time() - start} s")


if __name__ == "__main__":
    app.run(main)
