import numpy as np

from config import ForagingEnvConfig
from foraging_env import ForagingEnv
from heuristic_agent import OdorAwareWallFollowerController


def run_random_episode(env: ForagingEnv):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return {
        "success": info["success"],
        "steps": info["step_count"],
        "distance_to_food": info["distance_to_food"],
        "total_reward": total_reward,
    }


def run_heuristic_episode(env: ForagingEnv, controller: OdorAwareWallFollowerController):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = controller.choose_action(obs, info)
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

    random_env = ForagingEnv(config=config, render_mode=None)
    heuristic_env = ForagingEnv(config=config, render_mode=None)

    random_results = []
    for _ in range(num_episodes):
        random_results.append(run_random_episode(random_env))

    heuristic_results = []
    for _ in range(num_episodes):
        controller = OdorAwareWallFollowerController(follow_side="left")
        heuristic_results.append(run_heuristic_episode(heuristic_env, controller))

    random_env.close()
    heuristic_env.close()

    summarize(random_results, "Random Agent")
    summarize(heuristic_results, "Heuristic Agent")


if __name__ == "__main__":
    main()
