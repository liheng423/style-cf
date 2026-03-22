# Style-CF Project Docs

This documentation keeps only the critical information needed to answer three questions quickly:

1. What this project does
2. Where the main entry points are
3. How to run training, testing, and IDM calibration

## Overview

Style-CF is a car-following modeling and experiment project that includes:

- StyleCF (Transformer + style token)
- LSTM, Transfollower, and IDM baselines
- Training, testing, and IDM calibration workflows

## Key Modules

- `src/training.py`: package training entry
- `testing.py`: external testing entry script
- `src/testing.py`: testing and evaluation callable module
- `src/exps/configs.py`: core configs for train/test/calibration
- `src/exps/agent.py`: closed-loop rollout agent
- `src/exps/idm_calibrate.py`: IDM calibration with genetic algorithm

## Suggested Reading Order

1. [Quickstart](quickstart.md)
2. [Workflow](workflow.md)
3. [Configs](configs.md)
4. [API Reference](api.md)
