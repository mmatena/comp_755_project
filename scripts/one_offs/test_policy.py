from absl import app
from absl import flags
import gym
from pyvirtualdisplay import Display
import tensorflow as tf

from rl755.data.car_racing import encoded_rollouts
from rl755.data_gen import gym_rollouts
from rl755.models.car_racing import saved_models
from rl755.models.car_racing import policies

FLAGS = flags.FLAGS


def main(_):
    # It looks like OpenAI gym requires some sort of display, so we
    # have to fake one.
    display = Display(visible=0, size=(400, 300))
    display.start()

    encoder = saved_models.raw_rollout_vae_32ld()
    encoder.compile()
    model = saved_models.encoded_rollout_transformer()
    model.compile()

    policy = policies.RandomShootingPolicy(
        encoder=encoder, model=model, num_samples=128
    )

    env = gym.make("CarRacing-v0")

    rollout = gym_rollouts.single_rollout(env, policy=policy, max_steps=2000)

    print(f"Cumulative reward: {sum(rollout.reward_l)}")


if __name__ == "__main__":
    app.run(main)
