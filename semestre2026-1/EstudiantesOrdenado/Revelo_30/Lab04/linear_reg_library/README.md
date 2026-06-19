# linear_reg_library

`linear_reg_library` is a small Python package for fitting a simple linear regression model with a quadratic cost function and gradient descent.

## Installation with uv

From the `linear_reg_library` folder, run:

```bash
uv pip install .
```

## Quick usage

```python
import numpy as np

from linear_reg_lib import fit_linear_regression

x = np.arange(6, dtype=float)
y = np.arange(6, dtype=float)

model = fit_linear_regression(x, y, learning_rate=0.01, iterations=2000)

print(model["theta0"])
print(model["theta1"])
print(model["cost"])
```

## Public API

### `hypothesis(theta0, theta1, x)`

Computes the linear hypothesis:

```text
h(x) = theta0 + theta1 * x
```

### `compute_cost(theta0, theta1, x, y)`

Computes the quadratic cost function:

```text
J(theta0, theta1) = sum((h(x) - y)^2) / (2m)
```

### `gradient_descent(x, y, theta0=0.0, theta1=0.0, learning_rate=0.01, iterations=1000)`

Updates the model parameters iteratively and returns:

- final `theta0`
- final `theta1`
- `cost_history`

### `fit_linear_regression(x, y, theta0=0.0, theta1=0.0, learning_rate=0.01, iterations=1000)`

Fits the model to a dataset and returns a dictionary with:

- `theta0`
- `theta1`
- `cost`
- `cost_history`

## Example execution with the lab data

The lab uses an ideal dataset of the form:

```python
x = np.arange(6, dtype=float)
y = np.arange(6, dtype=float)
```

To run the included example:

```bash
uv run example_usage.py
```

Expected behavior:

- `theta0` should be close to `0`
- `theta1` should be close to `1`
- the final cost should be close to `0`

## Example install and run flow

```bash
cd semestre2026-1/Estudiantes/Revelo_30/Laboratorios/Laboratorio_4/linear_reg_library
uv pip install .
uv run example_usage.py
```
