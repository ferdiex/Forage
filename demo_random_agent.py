import time

from config import ForagingEnvConfig
from controllers.factory import make_controller
from foraging_env import ForagingEnv


def main():
    env = ForagingEnv(ForagingEnvConfig(), render_mode="human")
    controller = make_controller("random")

    try:
        for episode in range(5):
            controller.reset()
            obs, info = env.reset()
            done = False

            while not done:
                action = controller.act(obs, info)
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
