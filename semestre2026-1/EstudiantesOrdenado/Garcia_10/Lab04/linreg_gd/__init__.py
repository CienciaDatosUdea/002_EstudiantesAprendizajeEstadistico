"""
linreg_gd
---------
Librería minimalista para ajustar una regresión lineal simple mediante
función de coste cuadrática y gradiente descendente.

Uso rápido
----------
>>> import numpy as np
>>> from linreg_gd import fit, hypothesis, cost_function

>>> X = np.linspace(0, 1, 100)
>>> y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

>>> modelo = fit(X, y, alpha=0.1, max_iter=10000)
>>> y_pred  = hypothesis(modelo["theta0"], modelo["theta1"], X)
"""

from .core import hypothesis, cost_function, gradient_descent, fit

__version__ = "1.0.0"
__author__  = "Garcia_10"
__all__     = ["hypothesis", "cost_function", "gradient_descent", "fit"]