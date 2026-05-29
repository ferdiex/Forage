import time
import numpy as np

from foraging_env import ForagingEnv
from config import ForagingEnvConfig


class OdorAwareWallFollowerController:
    """
    A simple heuristic controller that combines:
    - stable wall-following
    - short-term stuck recovery
    - basic odor-based directional bias

    Main behavior:
    - avoid obstacles first
    - when odor improves, keep moving
    - when odor worsens, try a short exploratory turn
    - if stuck, reverse and recover
    """

    def __init__(self, follow_side: str = "left"):
        assert follow_side in ("left", "right")
        self.follow_side = follow_side

        self.turn_steps_remaining = 0
        self.turn_action = None

        self.reverse_steps_remaining = 0

        self.last_positions = []
        self.stuck_window = 15
        self.stuck_distance_threshold = 8.0

        self.prev_odor = None
        self.best_odor = 0.0

        self.explore_turn_steps_remaining = 0
        self.explore_turn_action = None

    def choose_action(self, obs: np.ndarray, info: dict) -> int:
        prox = obs[:8]
        odor = float(obs[8])

        front = prox[0]
        front_left = prox[1]
        left = prox[2]
        back_left = prox[3]
        back = prox[4]
        back_right = prox[5]
        right = prox[6]
        front_right = prox[7]

        position = info.get("robot_position", None)
        if position is not None:
            self._update_position_history(np.array(position, dtype=np.float32))

        if odor > self.best_odor:
            self.best_odor = odor

        # Recovery mode: reverse first.
        if self.reverse_steps_remaining > 0:
            self.reverse_steps_remaining -= 1
            self.prev_odor = odor
            return 3

        # Recovery mode: then keep turning for a few steps.
        if self.turn_steps_remaining > 0:
            self.turn_steps_remaining -= 1
            self.prev_odor = odor
            return self.turn_action

        # Odor exploration mode: commit briefly to one turn.
        if self.explore_turn_steps_remaining > 0:
            self.explore_turn_steps_remaining -= 1
            self.prev_odor = odor
            return self.explore_turn_action

        # If stuck, execute a short recovery.
        if self._looks_stuck():
            self.reverse_steps_remaining = 3
            self.turn_action = 1 if self.follow_side == "left" else 2
            self.turn_steps_remaining = 4
            self.prev_odor = odor
            return 3

        strong_front_block = 0.55
        moderate_front_block = 0.30
        side_block = 0.75
        wall_band = 0.20

        if self.follow_side == "left":
            preferred_turn = 1
            opposite_turn = 2
            follow_sensor = left
            front_corner = front_left
        else:
            preferred_turn = 2
            opposite_turn = 1
            follow_sensor = right
            front_corner = front_right

        # Highest priority: obstacle avoidance.
        if front > strong_front_block or front_corner > strong_front_block:
            self.turn_action = preferred_turn
            self.turn_steps_remaining = 3
            self.prev_odor = odor
            return self.turn_action

        if follow_sensor > side_block:
            self.turn_action = opposite_turn
            self.turn_steps_remaining = 2
            self.prev_odor = odor
            return self.turn_action

        # If odor is present, use it to bias local search.
        if odor > 0.0 and self.prev_odor is not None:
            odor_delta = odor - self.prev_odor

            # If odor is clearly improving and front is open, keep moving.
            if odor_delta > 0.01 and front < moderate_front_block:
                self.prev_odor = odor
                return 0

            # If odor is clearly worsening, try a short turn.
            if odor_delta < -0.01:
                self.explore_turn_action = preferred_turn
                self.explore_turn_steps_remaining = 2
                self.prev_odor = odor
                return self.explore_turn_action

        # If the followed side is too open, gently turn toward it.
        if 0.0 < follow_sensor < wall_band and front < moderate_front_block:
            self.turn_action = preferred_turn
            self.turn_steps_remaining = 1
            self.prev_odor = odor
            return self.turn_action

        # If front is clear, move forward.
        if front < moderate_front_block:
            self.prev_odor = odor
            return 0

        # Fallback: turn to preferred side.
        self.turn_action = preferred_turn
        self.turn_steps_remaining = 2
        self.prev_odor = odor
        return self.turn_action

    def _update_position_history(self, position: np.ndarray):
        self.last_positions.append(position)
        if len(self.last_positions) > self.stuck_window:
            self.last_positions.pop(0)

    def _looks_stuck(self) -> bool:
        if len(self.last_positions) < self.stuck_window:
            return False

        start = self.last_positions[0]
        end = self.last_positions[-1]
        displacement = np.linalg.norm(end - start)
        return displacement < self.stuck_distance_threshold


def main():
    env = ForagingEnv(ForagingEnvConfig(), render_mode="human")
    controller = OdorAwareWallFollowerController(follow_side="left")

    num_episodes = 5

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = controller.choose_action(obs, info)
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
