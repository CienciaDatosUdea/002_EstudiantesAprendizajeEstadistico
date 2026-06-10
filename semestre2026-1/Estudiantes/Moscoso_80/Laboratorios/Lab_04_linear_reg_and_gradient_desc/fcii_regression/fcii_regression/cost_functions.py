"""
Funciones de coste para regresión lineal.

Este módulo contiene las funciones de coste utilizadas para medir
el error del modelo de regresión lineal durante el entrenamiento.
"""

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE) entre valores reales y predichos.

    El MSE es una métrica de error que penaliza los errores grandes más que
    los pequeños debido al término cuadrático. Es diferenciable, lo que la
    hace ideal para optimización por gradiente descendente.

    Fórmula: MSE = (1/2m) * Σ(y_pred - y_true)²

    Nota: Se usa 1/2m en lugar de 1/m para simplificar la derivada
    durante el gradiente descendente (el 2 del cuadrado se cancela).

    Args:
        y_true (np.ndarray): Valores reales (etiquetas). Shape: (m,)
        y_pred (np.ndarray): Valores predichos por el modelo. Shape: (m,)

    Returns:
        float: Valor del error cuadrático medio.

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.0, 2.8])
        >>> mse = mean_squared_error(y_true, y_pred)
        >>> print(f"MSE: {mse:.4f}")
        MSE: 0.0083
    """
    m = len(y_true)
    return (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)


def mse_gradient(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calcula el gradiente del MSE respecto a los parámetros θ.

    Para regresión lineal h(x) = θ₀ + θ₁x, las derivadas parciales son:
        ∂J/∂θ₀ = (1/m) * Σ(h(x) - y)
        ∂J/∂θ₁ = (1/m) * Σ(h(x) - y) * x

    Args:
        X (np.ndarray): Features de entrada con columna de 1s. Shape: (m, 2)
        y_true (np.ndarray): Valores reales. Shape: (m,)
        y_pred (np.ndarray): Valores predichos. Shape: (m,)

    Returns:
        np.ndarray: Gradiente [∂J/∂θ₀, ∂J/∂θ₁]. Shape: (2,)
    """
    m = len(y_true)
    error = y_pred - y_true
    gradient = (1 / m) * X.T.dot(error)
    return gradient
