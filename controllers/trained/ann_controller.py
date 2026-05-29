from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseController


class ANNController(BaseController):
    """Minimal feedforward ANN controller.

    The network is a stack of linear layers with tanh activations on all hidden
    layers and a final linear output layer. The selected action is argmax(logits).
    A small action-commitment mechanism is used for turn actions so the controller
    does not oscillate left/right every frame when it encounters an obstacle.
    """

    family = "trained"
    controller_type = "ann"

    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str] = None,
        turn_commit_steps: int = 3,
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
        self._committed_action: Optional[int] = None
        self._commit_remaining = 0

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        if self._commit_remaining > 0 and self._committed_action is not None:
            self._commit_remaining -= 1
            return int(self._committed_action)

        x = np.asarray(obs, dtype=np.float32)

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i < len(self.weights) - 1:
                x = np.tanh(x)

        action = int(np.argmax(x))

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
