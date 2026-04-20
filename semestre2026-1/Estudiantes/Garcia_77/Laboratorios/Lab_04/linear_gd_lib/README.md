# linear-gd-lib

Pequena libreria de Python para ajustar una regresion lineal simple a partir de funcion de costo cuadratica y gradiente descendente.

## Estructura

```text
linear_gd_lib/
  pyproject.toml
  README.md
  src/
    linear_gd_lib/
      __init__.py
      core.py
```

## Instalacion local

Desde la carpeta `linear_gd_lib`:

```bash
pip install -e .
```

## Uso basico

```python
import numpy as np
from linear_gd_lib import fit_linear_gd

X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

model = fit_linear_gd(X, y, alpha=0.5, n_iter=20000, tol=1e-12)

print(model["theta0"], model["theta1"])
y_pred = model["predict"](X)
```

## API

- `linear_hypothesis(X, theta0, theta1)`
- `linear_cost(X, y, theta0, theta1)`
- `run_gradient_descent(X, y, alpha=0.1, n_iter=10000, tol=1e-12)`
- `fit_linear_gd(X, y, alpha=0.1, n_iter=10000, tol=1e-12)`

## Notas

- Esta version implementa regresion lineal simple (una sola variable `X`).
- El ajuste depende de la tasa de aprendizaje `alpha` y del criterio de parada `tol`.
