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
    """

    family = "trained"
    controller_type = "ann"

    def __init__(
        self,
        model: Any = None,
        model_path: Optional[str] = None,
    ):
        if model is None and model_path is None:
            raise ValueError("ANNController requires either 'model' or 'model_path'.")

        if model is None:
            model = self._load_model(model_path)

        self.weights = [np.asarray(layer, dtype=np.float32) for layer in model["weights"]]
        self.biases = [np.asarray(layer, dtype=np.float32) for layer in model["biases"]]

        if len(self.weights) != len(self.biases):
            raise ValueError("Model weights and biases must have the same number of layers.")

    def act(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        x = np.asarray(obs, dtype=np.float32)

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight + bias
            if i < len(self.weights) - 1:
                x = np.tanh(x)

        return int(np.argmax(x))

    def _load_model(self, model_path: str) -> Dict[str, Any]:
        path = Path(model_path)
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
