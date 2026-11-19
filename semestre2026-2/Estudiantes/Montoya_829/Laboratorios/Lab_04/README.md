# Regresion lineal por descenso por gradiente

Este paquete implementa una regresion lineal simple de una variable mediante la funcion de coste:

$$J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(\theta_0 + \theta_1x^{(i)} - y^{(i)})^2.$$

## Instalacion

Desde la carpeta `Lab_04`:

```bash
python -m pip install .
```

Para desarrollo editable:

```bash
python -m pip install -e .
```

## Uso

```python
import numpy as np

from regresion_lineal_gd import fit_linear_regression

x = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * x
resultado = fit_linear_regression(x, y, learning_rate=0.5)

print(resultado.intercept)
print(resultado.slope)
print(resultado.predict([0.0, 0.5, 1.0]))
```

El resultado incluye `intercept`, `slope`, `cost`, `iterations` e `history`.

## API

- `linear_hypothesis(x, intercept, slope)`: calcula las predicciones lineales.
- `cost_function(x, y, intercept, slope)`: calcula el coste cuadratico solicitado en el laboratorio.
- `gradient_descent(x, y, ...)`: ajusta los parametros con actualizaciones simultaneas.
- `fit_linear_regression(x, y, ...)`: funcion principal de ajuste.

## Ejemplo ejecutable

Tras instalar el paquete, ejecute:

```bash
python example.py
```

El ejemplo usa los datos definidos en el punto 12 del laboratorio e imprime los parametros, el coste y las iteraciones del ajuste.
