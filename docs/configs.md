# Configs

Config source files now use TOML and are loaded by one utility:

- loader: `src/utils/config_utils.py`
- data config: `src/exps/config_toml/datahandle.toml`
- training config: `src/exps/config_toml/train.toml`
- model config: `src/exps/config_toml/models.toml`
- testing config: `src/exps/config_toml/test.toml`

`src/exps/configs.py` remains the compatibility entry and re-exports resolved config objects.

## `style_data_config`

Training data build config:

- `seq_len` / `label_len` / `pred_len` / `stride`
- `x_groups`: encoder, decoder, and style feature groups
- `y_groups`: prediction target groups
- `dataset`: dataset class (resolved from symbol in TOML)

## `style_train_config`

Training hyperparameters:

- `num_epoch`
- `lr`
- `optim` (resolved from symbol in TOML)
- `loss_func` (resolved in loader)
- `device`
- `best_model_path`

## `lstm_data_config` / `lstm_model_config`

Data windows and model structure for the LSTM baseline.

## `idm_calibration_config`

IDM calibration parameters:

- `x_groups` / `y_groups`
- `downsample`
- `pred_horizon` / `historic_step` / `start_step`
- `loss` (resolved in loader)
- `device`
- `save_path`

## `test_config`

Testing and evaluation config:

- `datapath`
- `device`
- `style_window` / `test_window`
- `enabled_models` (optional)
- `style_token_seconds` / `style_token_mode` (optional)
- `output_dir` / `save_results` / `plot_results` (optional)
- per-model `*_agent` config (symbol names resolved by loader)

## Suggested Order

1. Edit TOML files first
2. Keep symbols consistent with loader maps in `src/utils/config_utils.py`
3. Run baseline train/test
4. Add env overrides only when needed
