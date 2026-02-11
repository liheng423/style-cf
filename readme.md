# Style-CF / StyleCF

Style-based car-following modeling and experiments. The project includes:
- Transformer-based StyleCF model (style embedding + following prediction)
- LSTM, Transfollower, and IDM baselines
- ZenTraffic trajectory processing and sample construction
- Training, testing, and IDM calibration examples

## Project Structure
- `src/exps/`: training, testing, models, loss, configs
- `src/exps/models/`: StyleTransformer, Transfollower, LSTM, IDM
- `src/exps/datahandle/`: datasets, features, filters, scalers
- `src/dataprocess/`: raw trajectory processing and extraction
- `training.py`: StyleCF training example
- `testing.py`: testing/inference example (incomplete)
- `idm_calibrate.py`: IDM calibration example
- `zen_data_example.py`: ZenTraffic raw data processing example
- `test/`: unit tests and test runner

## Requirements
Recommended Python 3.10+. Key dependencies (non-exhaustive):
- `torch`
- `numpy`
- `pandas`
- `scikit-learn`
- `tensordict`
- `tqdm`
- `tslearn`
- `sko` (for IDM GA calibration)
- `matplotlib` (for some visualizations)

You can add a `requirements.txt` or install missing packages as needed.

## Data 
The project uses ZenTraffic preprocessed data (`.npy`) or raw CSV in multiple places:
- Training/testing: `data_path` in `training.py` and `testing.py`
- Config: `test_config["datapath"]` in `src/exps/configs.py`
- Raw processing example: `zen_data_example.py`

These paths are currently hard-coded examples (some are Windows paths). Update them to match your local data locations.

### ZenTraffic Raw Data Processing
See `zen_data_example.py`. The pipeline includes:
1. Load raw CSV trajectories
2. Time parsing, Kalman filtering, indexing, cleaning
3. Extract vehicle trajectories and car-following segments
4. Build CF samples for training

## Train StyleCF
1. Update `data_path` in `training.py`
2. Run:

```bash
python training.py
```

Models are saved to `models/best-model-*.pth` (path generated in `src/exps/configs.py`).

## IDM Calibration
1. Update `data_path` in `idm_calibrate.py`
2. Run:

```bash
python idm_calibrate.py
```

Calibration config is in `idm_calibration_config` in `src/exps/configs.py`.

## Testing and Inference
Test runner is `test/test_run.py` using `unittest`:

```bash
python -m unittest test/test_run.py
```

Some tests are skipped and depend on dataset paths; enable or edit as needed.

## Configs and Tunables
- `src/exps/configs.py`: main training/testing/calibration configs
- `style_data_config`: sequence lengths, feature groups, slicing
- `style_train_config`: optimizer, lr, device
- `data_filter_config` / `filter_names`: sample filtering rules

## Notes
- Many paths are placeholders; update before training.
- `testing.py` is incomplete (currently stops at dataloader build).
- For GPU, install a CUDA-enabled PyTorch build.

## License
No license yet. Add a LICENSE file if you plan to release publicly.
