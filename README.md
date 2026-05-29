# ForagingEnv

Minimal 2D foraging environment for comparing MLP, LSTM, and Transformer policies under partial observability.

## Stack
- Python
- Gymnasium
- NumPy
- pygame

## Files
- `config.py`: environment parameters
- `foraging_env.py`: Gymnasium environment
- `demo_random_agent.py`: simple random-agent demo

## Install
```bash
pip install gymnasium numpy pygame
```

## Run demo
```bash
python demo_random_agent.py
```

## Current features
- 2D world
- one robot
- one food source
- obstacle field
- 8 proximity sensors
- 1 scalar odor signal
- sparse reward
- termination by success or timeout
- non-terminal collisions

## Next steps
- refine sensor model
- refine collision behavior
- add map templates
- add debug info
- add training script
- add heuristic controller