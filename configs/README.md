# Experiment configs

This directory contains JSON experiment files for the current controller families.

## Run examples

```bash
python run_experiment.py --config configs/random_baseline.json
python run_experiment.py --config configs/braitenberg_baseline.json
python run_experiment.py --config configs/heuristic_baseline.json
python run_experiment.py --config configs/finite_state_baseline.json
```

## Current schema

```json
{
  "experiment": {
    "name": "heuristic_baseline",
    "episodes": 30,
    "render": false,
    "sleep": 0.03
  },
  "environment": {},
  "controller": {
    "type": "heuristic",
    "params": {
      "follow_side": "left"
    }
  }
}
```
