import time

from foraging_env import ForagingEnv
from config import ForagingEnvConfig


def main():
    env = ForagingEnv(ForagingEnvConfig(), render_mode="human")
    obs, info = env.reset()

    try:
        for episode in range(5):
            obs, info = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                time.sleep(0.03)

            print(
                f"episode={episode} success={info['success']} "
                f"steps={info['step_count']} dist={info['distance_to_food']:.2f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()