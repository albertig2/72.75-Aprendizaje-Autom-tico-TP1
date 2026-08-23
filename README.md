# 72.75 - Aprendizaje Automático - TP1

Regresión lineal y polinómica sobre `insurance.csv`.

## Setup

Requiere [uv](https://docs.astral.sh/uv/). Python 3.14 (ver `.python-version`) lo
instala uv automáticamente si falta.

```sh
uv sync
```

Esto crea `.venv/` con las dependencias fijadas en `uv.lock`.

## Uso

```sh
uv run python data_prep.py
uv run python linear_regression.py
uv run python polynomial_regression.py
uv run python final_evaluation.py
```

`uv run` usa el entorno del proyecto sin necesidad de activarlo. Si preferís
activarlo: `source .venv/bin/activate`.

## Dependencias

Se declaran en `pyproject.toml`: `numpy`, `pandas`, `scikit-learn`.
Para agregar una: `uv add <paquete>` (actualiza `pyproject.toml` y `uv.lock`).
