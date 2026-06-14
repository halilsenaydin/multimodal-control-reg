# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CARLA autonomous driving control prediction. The model predicts `[brake, throttle, steer]` from CARLA simulator recordings. The research was conducted in two distinct phases:

- **Phase 1** — Tabular baseline: ego-state features only, trained with a sklearn MLP Regressor. Goal was to quantify how much predictive power the state variables alone carry.
- **Phase 2** — Hybrid deep model: raw LiDAR point clouds are converted to BEV images and fed into a pretrained CNN backbone with ConvLSTM for temporal context, while a parallel camera branch encodes RGB images using a pretrained vision backbone; both streams are fused with a GRU over ego-state features into a single prediction head.

## Commands

### Phase 1 — Tabular Baseline (sklearn MLP)

```bash
python base_model.py
```

### Phase 2 — Hybrid Deep Model (BEV + Ego-state)

```bash
# Step 1: Convert raw LiDAR and RGB images parquet → BEV .npy files (run once)
python utils/bev/lidar_to_bev.py
python utils/image/images_to_npy.py
python utils/image/images_to_npy_multi.py

# Step 2: Train the model
python main.py

# Step 3: Evaluate on test set (loads checkpoints/best_model.pt)
python evaluate.py
```

## Code Quality Standards

### Naming Conventions

- Variables, functions, modules: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_single_leading_underscore`

### Documentation

- Google Style Docstrings for all public methods and classes
- English docstrings
- Inline comments only for complex logic — not for self-explanatory code

### Type Hints

- Use type hints on every function signature, always specify return type
- Use built-in generics for Python 3.10+: `list[str]`, `dict[str, int]`
- Use `typing` module for complex types: `Optional`, `Union`

### SOLID & Design

- Single Responsibility: one class, one purpose
- Use Abstract Base Classes (`abc.ABC`) for interfaces
- Prefer dependency injection over hard-coded dependencies
- Prefer composition over inheritance
- No magic numbers — use named constants in `config.py`
- Catch specific exceptions, never use bare `except` blocks
- No commented-out code — delete it

### Research Code Balance

- Prefer clarity over abstraction in research/prototyping code
- Do not introduce design patterns unless the code clearly benefits from them
- Always preserve tensor shape comments in format `# [B, C, H, W]` — never remove them
