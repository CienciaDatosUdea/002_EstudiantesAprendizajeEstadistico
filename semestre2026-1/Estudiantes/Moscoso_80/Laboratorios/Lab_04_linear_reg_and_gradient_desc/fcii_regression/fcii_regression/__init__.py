"""
fcii_regression - Librería de Regresión Lineal con Gradiente Descendente

Esta librería implementa un modelo de regresión lineal simple optimizado
mediante el algoritmo de gradiente descendente, desarrollada como parte
del Laboratorio 4 del curso Fundamentos de Computación II.

Clases principales:
    - GradientDescent: Optimizador de gradiente descendente genérico.
    - LinearRegression: Modelo de regresión lineal (y = θ₀ + θ₁x).

Funciones de coste:
    - mean_squared_error: Error cuadrático medio (MSE).
    - mse_gradient: Gradiente del MSE.

Ejemplo de uso:
    >>> from fcii_regression import GradientDescent, LinearRegression
    >>> import numpy as np
    >>>
    >>> # Generar datos sintéticos
    >>> X = np.linspace(0, 10, 50)
    >>> y = 2.5 * X + 1.0 + np.random.randn(50) * 2
    >>>
    >>> # Configurar el optimizador
    >>> gd = GradientDescent(n_iterations=1000, learning_rate=0.01)
    >>>
    >>> # Crear y entrenar el modelo
    >>> model = LinearRegression(optimizer=gd)
    >>> model.fit(X, y)
    >>>
    >>> # Ver resultados
    >>> print(model.get_equation())
    >>> print(f"R² Score: {model.score(X, y):.4f}")
"""

__version__ = "1.0.0"
__author__ = "Estudiante FCII - UdeA"

from .gradient_descent import GradientDescent
from .linear_regression import LinearRegression
from .cost_functions import mean_squared_error, mse_gradient

__all__ = [
    "GradientDescent",
    "LinearRegression",
    "mean_squared_error",
    "mse_gradient",
]
