import json
from pathlib import Path
from typing import Any, Dict

from config import ForagingEnvConfig


DEFAULT_CONFIG = {
    "experiment": {
        "name": "default_experiment",
        "episodes": 10,
        "render": False,
        "sleep": 0.03,
    },
    "environment": {},
    "controller": {
        "type": "heuristic",
        "params": {
            "follow_side": "left",
        },
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_update(DEFAULT_CONFIG, user_config)


def build_env_config(config_dict: Dict[str, Any]) -> ForagingEnvConfig:
    env_dict = config_dict.get("environment", {})
    return ForagingEnvConfig.from_dict(env_dict)
