import time

from config import ForagingEnvConfig
from controllers.factory import make_controller
from foraging_env import ForagingEnv


def main():
    env = ForagingEnv(ForagingEnvConfig(), render_mode="human")
    controller = make_controller("heuristic", follow_side="left")

    num_episodes = 5

    try:
        for episode in range(num_episodes):
            controller.reset()
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = controller.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                time.sleep(0.03)

            print(
                f"episode={episode} "
                f"success={info['success']} "
                f"steps={info['step_count']} "
                f"dist={info['distance_to_food']:.2f} "
                f"total_reward={total_reward:.1f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
