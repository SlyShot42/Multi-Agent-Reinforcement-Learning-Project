# Multi-Agent Extraction RL Prototype

A lightweight multi-agent grid game built on `pygame`, where each [`resources.agent.Agent`](resources/agent.py) learns to navigate toward its assigned extraction target [`resources.end_pnt.EndPnt`](resources/end_pnt.py) while avoiding collisions. Orchestration, reward logic, and training hooks live in [`resources.main_game.MainGame`](resources/main_game.py), and a simple DQN backbone is provided via [`resources.model.DQN`](resources/model.py).

## Core Loop

1. [`main.py`](main.py) boots a [`MainGame`](resources/main_game.py) with two agents.
2. Each tick, `MainGame.step` applies a list of [`resources.agent.Action`](resources/agent.py) values, updates board occupancy, assigns rewards ($-10$ for collisions/boundaries, $+10$ for successful extraction), and refreshes agent states.
3. `MainGame.reset` respawns finished agents, maintains short/long memory buffers, and keeps per-agent score/record counters for training diagnostics.

## File Layout

- `resources/agent.py` – Agent body kinematics, ε-greedy policy helpers, replay memory, and 11-dimensional state encoder.
- `resources/main_game.py` – World generation, collision resolution, reward scoring, draw loop, and RL training harness.
- `resources/model.py` – Torch-based multi-layer perceptron implementing the Q-network used by agents.
- `resources/game_obj.py`, `resources/end_pnt.py`, `resources/config.py` – Shared geometry and rendering utilities.
- `tests/` – PyTest suites covering movement, collision-free entity spawning, agent state encoding, and reward/done bookkeeping.

## Getting Started

```bash
pip install -r requirements.txt  # ensure pygame, torch, pytest, icecream, termcolor, etc.
python main.py                   # trains then starts the interactive loop
```

## Tests

```bash
pytest
```

The suites in [tests/step_test.py](tests/step_test.py), [tests/agent_update_state_test.py](tests/agent_update_state_test.py), [tests/generate_entities_test.py](tests/generate_entities_test.py), and [tests/move_agent_test.py](tests/move_agent_test.py) validate rewards, state encodings, spawn safety, and directional moves.

## Notes

- Display logic depends on a `pygame` window sized by the grid constants in [`resources.config`](resources/config.py).
- Replay training is stubbed: plug in a trainer/model instance on each `Agent` before calling `MainGame.train` for full DQN updates.
