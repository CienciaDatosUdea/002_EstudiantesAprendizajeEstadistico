"""
linreg_gd — Regresión lineal mediante función de coste y gradiente descendente.

Módulo que implementa un modelo de regresión lineal simple usando
el método de gradiente descendente para minimizar la función de coste MSE.
"""

from .core import hypothesis, cost_function, gradient_descent, fit

__version__ = "0.1.0"
__all__ = ["hypothesis", "cost_function", "gradient_descent", "fit"]
