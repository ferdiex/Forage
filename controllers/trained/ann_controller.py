from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseController


class ANNController(BaseController):
    """Feedforward ANN controller with an explicit unstuck routine."""

    family = "trained"
    controller_type = "ann"

    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str] = None,
        turn_commit_steps: int = 3,
        obstacle_threshold: float = 0.6,
        critical_threshold: float = 0.85,
        clear_threshold: float = 0.25,
        reverse_steps: int = 2,
        turn_steps: int = 8,
        forward_steps: int = 4,
    ):
        if model is None and model_path is None:
            raise ValueError("ANNController requires either 'model' or 'model_path'.")

        if model is None:
            model = self._load_model(model_path)

        self.weights = [np.asarray(layer, dtype=np.float32) for layer in model["weights"]]
        self.biases = [np.asarray(layer, dtype=np.float32) for layer in model["biases"]]

        if len(self.weights) != len(self.biases):
            raise ValueError("Model weights and biases must have the same number of layers.")

        self.turn_commit_steps = max(0, int(turn_commit_steps))
        self.obstacle_threshold = float(obstacle_threshold)
        self.critical_threshold = float(critical_threshold)
        self.clear_threshold = float(clear_threshold)

        self.reverse_steps = max(0, int(reverse_steps))
        self.turn_steps = max(0, int(turn_steps))
        self.forward_steps = max(0, int(forward_steps))

        self._committed_action: Optional[int] = None
        self._commit_remaining = 0

        self._unstuck_plan: List[int] = []

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        obs = np.asarray(obs, dtype=np.float32)

        front = float(obs[0])
        front_left = float(obs[1])
        front_right = float(obs[7])

        if self._unstuck_plan:
            if front <= self.clear_threshold and self._unstuck_plan[0] == 0:
                self._unstuck_plan.clear()
            else:
                return int(self._unstuck_plan.pop(0))

        if self._commit_remaining > 0 and self._committed_action is not None:
            self._commit_remaining -= 1
            return int(self._committed_action)

        if front >= self.critical_threshold:
            turn_action = 1 if front_left < front_right else 2
            self._unstuck_plan = (
                [3] * self.reverse_steps
                + [turn_action] * self.turn_steps
                + [0] * self.forward_steps
            )
            return int(self._unstuck_plan.pop(0))

        x = obs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i < len(self.weights) - 1:
                x = np.tanh(x)

        action = int(np.argmax(x))

        if front >= self.obstacle_threshold and action == 0:
            turn_action = 1 if front_left < front_right else 2
            self._unstuck_plan = [turn_action] * self.turn_steps + [0] * self.forward_steps
            return int(self._unstuck_plan.pop(0))

        if action in (1, 2) and self.turn_commit_steps > 0:
            self._committed_action = action
            self._commit_remaining = self.turn_commit_steps - 1
        else:
            self._committed_action = None
            self._commit_remaining = 0

        return action

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        path = Path(model_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
