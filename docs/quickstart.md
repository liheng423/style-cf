# Quickstart

## 1. Install doc dependencies

```bash
pip install -r docs/requirements.txt
```

## 2. Preview docs locally

```bash
mkdocs serve
```

Open the local URL from terminal output (default: `http://127.0.0.1:8000`).

## 3. Build static site

```bash
mkdocs build
```

Build output is in `site_docs/`. Open `site_docs/index.html` in a browser.

## 4. Run key project entries

Train:

```bash
python -m src.training
```

Test:

```bash
python testing.py
```

IDM calibration:

```bash
python idm_calibrate.py
```

## 5. Paths you must update before running

- Training dataset path: `_dataset()` in `src/training.py`
- Testing dataset path: `test_config["datapath"]` in `src/exps/configs.py` or env var `ZEN_DATA_PATH`
- Calibration dataset path: `_dataset()` in `idm_calibrate.py`
