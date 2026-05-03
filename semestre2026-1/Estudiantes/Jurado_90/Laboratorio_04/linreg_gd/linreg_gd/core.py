"""
Funciones núcleo de regresión lineal con gradiente descendente.
"""

import numpy as np


def hypothesis(X, theta0, theta1):
    """
    Calcula la hipótesis lineal h(X) = θ₀ + θ₁·X.

    Parameters
    ----------
    X : array-like
        Vector de características (m muestras).
    theta0 : float
        Parámetro de sesgo (intercepto).
    theta1 : float
        Parámetro de pendiente.

    Returns
    -------
    numpy.ndarray
        Predicciones del modelo lineal.
    """
    X = np.asarray(X, dtype=float)
    return theta0 + theta1 * X


def cost_function(X, y, theta0, theta1):
    """
    Calcula la función de coste cuadrática (MSE/2).

    J(θ₀, θ₁) = (1/2m) · Σᵢ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

    Parameters
    ----------
    X : array-like
        Vector de características (m muestras).
    y : array-like
        Vector de valores objetivo (m muestras).
    theta0 : float
        Parámetro de sesgo.
    theta1 : float
        Parámetro de pendiente.

    Returns
    -------
    float
        Valor de la función de coste.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    predictions = hypothesis(X, theta0, theta1)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


def gradient_descent(
    X, y, theta0=0.0, theta1=0.0, alpha=0.01, epsilon=1e-6, max_iter=50000
):
    """
    Ejecuta el gradiente descendente para regresión lineal.

    Actualiza simultáneamente θ₀ y θ₁ hasta que la variación
    sea menor que epsilon o se alcance max_iter iteraciones.

    Parameters
    ----------
    X : array-like
        Vector de características (m muestras).
    y : array-like
        Vector de valores objetivo (m muestras).
    theta0 : float, optional
        Valor inicial de θ₀ (default 0.0).
    theta1 : float, optional
        Valor inicial de θ₁ (default 0.0).
    alpha : float, optional
        Tasa de aprendizaje (default 0.01).
    epsilon : float, optional
        Tolerancia de convergencia (default 1e-6).
    max_iter : int, optional
        Número máximo de iteraciones (default 50000).

    Returns
    -------
    tuple
        (theta0, theta1, cost_history, n_iterations)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    m = len(y)
    cost_history = []

    for i in range(max_iter):
        predictions = theta0 + theta1 * X
        error = predictions - y

        d_theta0 = (1 / m) * np.sum(error)
        d_theta1 = (1 / m) * np.sum(error * X)

        theta0_new = theta0 - alpha * d_theta0
        theta1_new = theta1 - alpha * d_theta1

        cost_history.append(cost_function(X, y, theta0, theta1))

        if abs(theta0_new - theta0) < epsilon and abs(theta1_new - theta1) < epsilon:
            theta0, theta1 = theta0_new, theta1_new
            break

        theta0, theta1 = theta0_new, theta1_new

    return theta0, theta1, cost_history, len(cost_history)


def fit(X, y, alpha=0.1, epsilon=1e-6, max_iter=50000):
    """
    Ajusta un modelo de regresión lineal a los datos usando gradiente descendente.

    Función principal que encapsula todo el flujo: recibe datos y retorna
    los parámetros óptimos junto con métricas de entrenamiento.

    Parameters
    ----------
    X : array-like
        Vector de características (m muestras).
    y : array-like
        Vector de valores objetivo (m muestras).
    alpha : float, optional
        Tasa de aprendizaje (default 0.1).
    epsilon : float, optional
        Tolerancia de convergencia (default 1e-6).
    max_iter : int, optional
        Número máximo de iteraciones (default 50000).

    Returns
    -------
    dict
        Diccionario con:
        - 'theta0': intercepto óptimo
        - 'theta1': pendiente óptima
        - 'cost_history': lista de valores de J por iteración
        - 'n_iterations': número de iteraciones realizadas
        - 'final_cost': valor final de J

    Example
    -------
    >>> import numpy as np
    >>> from linreg_gd import fit, hypothesis
    >>> X = np.linspace(0, 1, 100)
    >>> y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)
    >>> result = fit(X, y, alpha=0.5)
    >>> print(f"θ₀={result['theta0']:.4f}, θ₁={result['theta1']:.4f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    theta0, theta1, cost_history, n_iter = gradient_descent(
        X, y, alpha=alpha, epsilon=epsilon, max_iter=max_iter
    )

    return {
        "theta0": theta0,
        "theta1": theta1,
        "cost_history": cost_history,
        "n_iterations": n_iter,
        "final_cost": cost_function(X, y, theta0, theta1),
    }
