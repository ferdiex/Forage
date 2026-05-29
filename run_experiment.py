import argparse
import time

import numpy as np

from controllers.factory import make_controller
from experiment_config import build_env_config, load_experiment_config
from foraging_env import ForagingEnv


def run_episode(env: ForagingEnv, controller):
    controller.reset()
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "success": info["success"],
        "steps": info["step_count"],
        "distance_to_food": info["distance_to_food"],
        "total_reward": total_reward,
    }


def run_episode_with_render(env: ForagingEnv, controller, sleep_time: float):
    controller.reset()
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = controller.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(sleep_time)

    return {
        "success": info["success"],
        "steps": info["step_count"],
        "distance_to_food": info["distance_to_food"],
        "total_reward": total_reward,
    }


def summarize(results, experiment_name: str, controller_type: str):
    success_rate = np.mean([r["success"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_distance = np.mean([r["distance_to_food"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])

    print(f"=== {experiment_name} ===")
    print(f"controller: {controller_type}")
    print(f"episodes: {len(results)}")
    print(f"success_rate: {success_rate:.3f}")
    print(f"avg_steps: {avg_steps:.2f}")
    print(f"avg_final_distance: {avg_distance:.2f}")
    print(f"avg_total_reward: {avg_reward:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Forage experiment from JSON config.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment JSON config.")
    return parser.parse_args()


def main():
    args = parse_args()
    experiment = load_experiment_config(args.config)

    experiment_meta = experiment["experiment"]
    controller_cfg = experiment["controller"]
    controller_type = controller_cfg["type"]
    controller_params = controller_cfg.get("params", {})

    env_config = build_env_config(experiment)
    render_mode = "human" if experiment_meta.get("render", False) else None
    sleep_time = float(experiment_meta.get("sleep", 0.03))
    num_episodes = int(experiment_meta.get("episodes", 10))

    env = ForagingEnv(config=env_config, render_mode=render_mode)
    controller = make_controller(controller_type, **controller_params)

    results = []
    try:
        for _ in range(num_episodes):
            if render_mode == "human":
                result = run_episode_with_render(env, controller, sleep_time)
            else:
                result = run_episode(env, controller)
            results.append(result)
    finally:
        env.close()

    summarize(results, experiment_meta.get("name", "unnamed_experiment"), controller_type)


if __name__ == "__main__":
    main()
