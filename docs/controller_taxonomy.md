# Controller taxonomy

This repository organizes controllers into two top-level families:

- `designed`
- `trained`

## Designed controllers

### `random`
Minimal baseline with no task structure.

### `braitenberg`
Reactive obstacle-avoidance controller with very limited competence.
It is mainly useful as an illustrative embodied-reactive baseline.

### `finite_state`
Simple stateful controller with `explore`, `avoid`, and `recover` modes.
It currently performs as an intermediate designed baseline.

### `heuristic`
Current strongest hand-designed baseline in the repository.
It combines wall-following, stuck recovery, and local odor-aware bias.

## Trained controllers

### `ann`
Now supports a minimal feedforward inference path using JSON-defined weights.
This is the first functional trained-controller slot.

### `lstm`
Placeholder for future recurrent controller experiments.

### `transformer`
Placeholder for future transformer-based policy experiments.

## Experiment workflow

The repository now includes a JSON-based experiment workflow:

- `configs/*.json` define experiment settings
- `run_experiment.py` loads a config and runs the selected controller
- controller parameters are passed through the shared controller factory

## Current qualitative ranking

At the current stage, the designed-controller baselines are roughly:

1. `heuristic` — strongest manual baseline
2. `finite_state` — intermediate stateful baseline
3. `braitenberg` — reactive illustrative baseline
4. `random` — trivial baseline

## Planned next layers

- richer ANN experiments and weight generation
- training code
- optimizers and losses
- recurrent and transformer-based trained controllers
