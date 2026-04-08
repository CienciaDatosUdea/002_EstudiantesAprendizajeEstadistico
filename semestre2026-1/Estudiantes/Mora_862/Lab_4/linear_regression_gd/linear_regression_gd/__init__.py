"""
Librería Linear Regression with Gradient Descent

Una librería simple pero completa para regresión lineal univariada
usando gradiente descendente como algoritmo de optimización.

Uso básico:
    >>> from linear_regression_gd import LinearRegression
    >>> import numpy as np
    >>> 
    >>> # Crear datos
    >>> X = np.array([1, 2, 3, 4, 5], dtype=float)
    >>> y = np.array([2, 4, 5, 4, 5], dtype=float)
    >>> 
    >>> # Crear y entrenar modelo
    >>> model = LinearRegression()
    >>> model.fit(X, y, learning_rate=0.01, verbose=True)
    >>> 
    >>> # Hacer predicciones
    >>> predictions = model.predict(X)
    >>> 
    >>> # Evaluar modelo
    >>> r2 = model.score(X, y)
    >>> print(f"R² = {r2:.4f}")
    >>> 
    >>> # Visualizar resultados
    >>> model.plot_results(X, y)

Clases principales:
    - LinearRegression: Modelo de regresión lineal con gradiente descendente

Funciones de utilidad:
    - linear_hypothesis: Calcula predicciones
    - compute_cost: Calcula la función de coste

Autor: Estudiante de Física Computacional II
Universidad de Antioquia - 2026
"""

from .regression import LinearRegression, linear_hypothesis, compute_cost

__version__ = "1.0.0"
__author__ = "Estudiante FC II"
__all__ = ['LinearRegression', 'linear_hypothesis', 'compute_cost']
