import numpy as np

from config import ForagingEnvConfig
from controllers.factory import make_controller
from foraging_env import ForagingEnv


EPISODE_CONTROLLERS = [
    ("random", {}, "Random Agent"),
    ("braitenberg", {}, "Braitenberg Agent"),
    ("heuristic", {"follow_side": "left"}, "Heuristic Agent"),
    ("finite_state", {"follow_side": "left"}, "Finite State Agent"),
]


def run_episode(env: ForagingEnv, controller_type: str, controller_kwargs: dict):
    controller = make_controller(controller_type, **controller_kwargs)
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


def summarize(results, name: str):
    success_rate = np.mean([r["success"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_distance = np.mean([r["distance_to_food"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])

    print(f"=== {name} ===")
    print(f"episodes: {len(results)}")
    print(f"success_rate: {success_rate:.3f}")
    print(f"avg_steps: {avg_steps:.2f}")
    print(f"avg_final_distance: {avg_distance:.2f}")
    print(f"avg_total_reward: {avg_reward:.2f}")
    print()


def main():
    num_episodes = 30
    config = ForagingEnvConfig()

    for controller_type, controller_kwargs, label in EPISODE_CONTROLLERS:
        env = ForagingEnv(config=config, render_mode=None)
        results = []

        for _ in range(num_episodes):
            results.append(run_episode(env, controller_type, controller_kwargs))

        env.close()
        summarize(results, label)


if __name__ == "__main__":
    main()
