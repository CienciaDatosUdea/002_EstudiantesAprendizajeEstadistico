# mireglineal

Librería simple para regresión lineal con mínimos cuadrados y gradiente descendente.

## Instalación

pip install .

## Uso
```python
from mireglineal.regresion import ajustar
theta_0, theta_1 = ajustar(X, y, metodo='gradiente', alpha=0.1)