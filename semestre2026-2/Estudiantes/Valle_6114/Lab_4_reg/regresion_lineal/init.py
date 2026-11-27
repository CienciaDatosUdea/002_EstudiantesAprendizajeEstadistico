"""
Librería de Regresión Lineal por Descenso de Gradiente
"""

from .core import hipotesis, cost_function, gradient_descent, fit_linear_regression, predict

__all__ = [
    'hipotesis',
    'cost_function',
    'gradient_descent',
    'fit_linear_regression',
    'predict'
]