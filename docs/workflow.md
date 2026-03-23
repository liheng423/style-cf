# Workflow

## Training Workflow (StyleCF)

1. `src/training.py::_dataset()` loads ZenTraffic data
2. `build_style_loader(...)` builds dataset and DataLoaders
3. `train_stylecf(...)` trains and saves the best model

Core code:

- `src/exps/train/model_trainer.py`
- `src/exps/configs.py`

## Testing Workflow (Multi-model Evaluation)

Entry point: `testing.py::main()`

1. `_build_options()` reads config and env vars
2. `_build_eval_bundle()` splits style and test windows
3. `run_testing()` evaluates each enabled model
4. `summarize_results()` aggregates metrics
5. `save_outputs()` optionally saves CSV and plots

Common model names:

- `stylecf`
- `transformer`
- `idm`
- `lstm` (if configured)

## IDM Calibration Workflow

Entry point: `src/exps/idm_calibrate.py::calibrate_idm(...)`

1. Build `IDMDataset` for each sampled ID
2. Run `calibrate_idm_genetic(...)`
3. Use `_fitness_function(...)` for closed-loop evaluation through `Agent`
4. Save per-ID best parameters and losses

## Platoon Simulation Workflow

Entry point: `platoon_simulation.py` or `src/platoon/simulation.py::main()`

1. `src/platoon/config_loader.py` loads TOML defaults and active override
2. `src/platoon/builder.py` builds style token bank and simulation environment
3. `src/platoon/runner.py` executes configured experiment groups
4. `src/platoon/evaluator.py` computes delay, standstill, wave, FD, and fuel metrics
5. Outputs are saved to `simulation_config.output_dir` as `summary.csv`, per-group json, and `anova.json`

## Key Environment Variables

- `ZEN_DATA_PATH`: override testing dataset path
- `TEST_HEAD`: cap sample count for quick runs
- `TEST_PLOT`: enable plotting (`1/true`)
