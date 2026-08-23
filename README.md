# 72.75 - Aprendizaje Automático - TP1

Linear and polynomial regression on `insurance.csv`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 (see `.python-version`) is
installed automatically by uv if missing.

```sh
uv sync
```

This creates `.venv/` with the dependencies pinned in `uv.lock`.

## Usage

```sh
uv run python data_prep.py
uv run python linear_regression.py
uv run python polynomial_regression.py
uv run python final_evaluation.py
```

`uv run` uses the project environment without activating it. If you prefer to
activate it: `source .venv/bin/activate`.

## Dependencies

Declared in `pyproject.toml`: `numpy`, `pandas`, `scikit-learn`.
To add one: `uv add <package>` (updates both `pyproject.toml` and `uv.lock`).
