# Forage

A small Python repository for experimenting with foraging controllers in a 2D environment.

## Current controller taxonomy

Controllers are organized into two top-level families:

- `designed`
- `trained`

### Designed controllers
- `random`
- `braitenberg`
- `heuristic`
- `finite_state`

### Trained controllers
- `ann`
- `lstm`
- `transformer`

## Current baseline status

- `heuristic` is currently the strongest manual baseline
- `finite_state` is an intermediate stateful baseline
- `braitenberg` is a reactive illustrative baseline
- `random` is the trivial baseline

## JSON experiment runner

Experiments can now be configured with JSON files and run through a shared runner.

Examples:

```bash
python run_experiment.py --config configs/random_baseline.json
python run_experiment.py --config configs/braitenberg_baseline.json
python run_experiment.py --config configs/heuristic_baseline.json
python run_experiment.py --config configs/finite_state_baseline.json
```

## Minimal ANN controller

The repository now includes a minimal feedforward ANN controller that can:

- load weights and biases from JSON
- run feedforward inference
- choose a discrete action from logits

You can generate new ANN model files with:

```bash
python scripts/init_ann_model.py --output models/ann_test.json --layers 9 8 4 --scale 0.1 --seed 0
```

### ANN experiment status

The ANN-based obstacle-avoidance experiments are currently **frozen as an exploratory baseline**.

What was tried:
- manual ANN weight tuning in `models/ann_test.json`
- render and non-render experiment configs
- turn commitment in `controllers/trained/ann_controller.py`
- obstacle safety overrides
- explicit unstuck routines

Current outcome:
- the ANN can move forward in open space
- it still gets stuck oscillating left/right near obstacles
- it is not yet a reliable obstacle-avoidance controller

Recommended next step:
- keep this ANN path as a reference baseline
- implement a robust heuristic obstacle-avoidance controller for comparison
- revisit ANN later with training, memory, or imitation learning

Relevant files:
- `controllers/trained/ann_controller.py`
- `models/ann_test.json`
- `configs/ann_test.json`
- `configs/ann_test_render.json`

## Next steps

- implement a stronger heuristic obstacle-avoidance baseline
- revisit ANN experiments later with training or memory
- add LSTM and transformer implementations
