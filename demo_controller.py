import argparse
import time

from config import ForagingEnvConfig
from controllers.factory import make_controller
from foraging_env import ForagingEnv


DEFAULT_CONTROLLER_KWARGS = {
    "random": {},
    "braitenberg": {},
    "heuristic": {"follow_side": "left"},
    "finite_state": {"follow_side": "left"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a controller demo in the Forage environment.")
    parser.add_argument(
        "--controller",
        type=str,
        default="heuristic",
        choices=["random", "braitenberg", "heuristic", "finite_state"],
        help="Controller type to run.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run.")
    parser.add_argument("--sleep", type=float, default=0.03, help="Delay between rendered steps.")
    return parser.parse_args()


def main():
    args = parse_args()

    env = ForagingEnv(ForagingEnvConfig(), render_mode="human")
    controller = make_controller(args.controller, **DEFAULT_CONTROLLER_KWARGS[args.controller])

    try:
        for episode in range(args.episodes):
            controller.reset()
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = controller.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                time.sleep(args.sleep)

            print(
                f"episode={episode} "
                f"controller={args.controller} "
                f"success={info['success']} "
                f"steps={info['step_count']} "
                f"dist={info['distance_to_food']:.2f} "
                f"total_reward={total_reward:.1f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
