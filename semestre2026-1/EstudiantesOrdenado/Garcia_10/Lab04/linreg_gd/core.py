"""
core.py
-------
Implementación de regresión lineal mediante función de coste y gradiente descendente.

Funciones disponibles:
    - hypothesis      : Calcula la hipótesis lineal h(X) = θ₀ + θ₁·X
    - cost_function   : Calcula la función de coste cuadrática J(θ₀, θ₁)
    - gradient_descent: Ejecuta el algoritmo de gradiente descendente
    - fit             : Función principal — ajusta el modelo a los datos

Basado en el Laboratorio 4 de Aprendizaje Estadístico — UdeA 2026-1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Hipótesis lineal
# ---------------------------------------------------------------------------

def hypothesis(theta0: float, theta1: float, X: np.ndarray) -> np.ndarray:
    """
    Calcula la hipótesis del modelo de regresión lineal.

    El modelo lineal predice los valores de salida como:
        h(X) = θ₀ + θ₁·X

    Parámetros
    ----------
    theta0 : float
        Término independiente (intercepto) del modelo.
    theta1 : float
        Pendiente (coeficiente) del modelo.
    X : np.ndarray
        Array 1-D con los valores de la variable predictora.

    Retorna
    -------
    np.ndarray
        Array 1-D con las predicciones ŷ para cada valor de X.

    Ejemplo
    -------
    >>> import numpy as np
    >>> X = np.array([0.0, 0.5, 1.0])
    >>> hypothesis(0.2, 0.3, X)
    array([0.2 , 0.35, 0.5 ])
    """
    return theta0 + theta1 * X


# ---------------------------------------------------------------------------
# 2. Función de coste (MSE con factor 1/2)
# ---------------------------------------------------------------------------

def cost_function(theta0: float, theta1: float,
                  X: np.ndarray, y: np.ndarray) -> float:
    """
    Calcula la función de coste cuadrática (MSE escalado) para la regresión lineal.

    La función de coste mide qué tan lejos están las predicciones del modelo
    respecto a los valores reales:

        J(θ₀, θ₁) = (1 / 2m) · Σᵢ (h(xᵢ) − yᵢ)²

    El factor 1/2 es convencional en optimización: simplifica la derivada
    al aplicar el gradiente descendente.

    Parámetros
    ----------
    theta0 : float
        Término independiente (intercepto) del modelo.
    theta1 : float
        Pendiente (coeficiente) del modelo.
    X : np.ndarray
        Array 1-D con los valores de la variable predictora.
    y : np.ndarray
        Array 1-D con los valores reales (etiquetas) de entrenamiento.

    Retorna
    -------
    float
        Valor escalar de la función de coste J(θ₀, θ₁).

    Lanza
    -----
    ValueError
        Si X e y no tienen el mismo número de elementos.

    Ejemplo
    -------
    >>> import numpy as np
    >>> X = np.array([0.0, 0.5, 1.0])
    >>> y = np.array([0.2, 0.35, 0.5])
    >>> cost_function(0.2, 0.3, X, y)
    0.0
    """
    if len(X) != len(y):
        raise ValueError(
            f"X e y deben tener el mismo tamaño. "
            f"Recibidos: len(X)={len(X)}, len(y)={len(y)}"
        )

    m = len(y)
    errores = hypothesis(theta0, theta1, X) - y
    return float(np.sum(errores ** 2) / (2 * m))


# ---------------------------------------------------------------------------
# 3. Gradiente descendente
# ---------------------------------------------------------------------------

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    theta0_init: float = 0.0,
    theta1_init: float = 0.0,
    alpha: float = 0.01,
    max_iter: int = 10_000,
    epsilon: float = 1e-6,
) -> dict:
    """
    Ejecuta el algoritmo de gradiente descendente para minimizar J(θ₀, θ₁).

    En cada iteración actualiza simultáneamente los dos parámetros usando
    las derivadas parciales de la función de coste:

        θ₀ := θ₀ − α · (1/m) · Σᵢ (h(xᵢ) − yᵢ)
        θ₁ := θ₁ − α · (1/m) · Σᵢ (h(xᵢ) − yᵢ) · xᵢ

    El algoritmo se detiene cuando el desplazamiento en el espacio de parámetros
    es menor que `epsilon` o se alcanza `max_iter`.

    Parámetros
    ----------
    X : np.ndarray
        Array 1-D con los valores de la variable predictora.
    y : np.ndarray
        Array 1-D con los valores reales de entrenamiento.
    theta0_init : float, opcional
        Valor inicial de θ₀. Por defecto 0.0.
    theta1_init : float, opcional
        Valor inicial de θ₁. Por defecto 0.0.
    alpha : float, opcional
        Tasa de aprendizaje (learning rate). Por defecto 0.01.
        Valores muy grandes pueden causar divergencia; valores muy pequeños
        hacen la convergencia muy lenta.
    max_iter : int, opcional
        Número máximo de iteraciones. Por defecto 10 000.
    epsilon : float, opcional
        Umbral de convergencia. El algoritmo para cuando el cambio en θ
        es menor que este valor. Por defecto 1e-6.

    Retorna
    -------
    dict con las siguientes claves:
        theta0        : float  — Valor óptimo encontrado para θ₀.
        theta1        : float  — Valor óptimo encontrado para θ₁.
        cost_history  : list   — Historial de J en cada iteración.
        theta0_history: list   — Historial de θ₀ en cada iteración.
        theta1_history: list   — Historial de θ₁ en cada iteración.
        iterations    : int    — Número de iteraciones ejecutadas.
        converged     : bool   — True si el algoritmo convergió antes de max_iter.

    Ejemplo
    -------
    >>> import numpy as np
    >>> X = np.linspace(0, 1, 50)
    >>> y = 0.2 + 0.3 * X
    >>> result = gradient_descent(X, y, alpha=0.1, max_iter=5000)
    >>> round(result["theta0"], 3), round(result["theta1"], 3)
    (0.2, 0.3)
    """
    m = len(y)
    theta0 = float(theta0_init)
    theta1 = float(theta1_init)

    cost_history   = []
    theta0_history = []
    theta1_history = []
    converged      = False

    for i in range(max_iter):
        predicciones = hypothesis(theta0, theta1, X)
        errores      = predicciones - y

        # Gradientes (derivadas parciales de J)
        grad_theta0 = (1 / m) * np.sum(errores)
        grad_theta1 = (1 / m) * np.sum(errores * X)

        # Actualización simultánea
        new_theta0 = theta0 - alpha * grad_theta0
        new_theta1 = theta1 - alpha * grad_theta1

        # Guardar historial antes de actualizar
        theta0_history.append(theta0)
        theta1_history.append(theta1)
        cost_history.append(cost_function(theta0, theta1, X, y))

        # Criterio de convergencia: norma del paso en el espacio de parámetros
        desplazamiento = np.sqrt((new_theta0 - theta0) ** 2 +
                                 (new_theta1 - theta1) ** 2)

        theta0, theta1 = new_theta0, new_theta1

        if desplazamiento < epsilon:
            converged = True
            break

    return {
        "theta0"        : theta0,
        "theta1"        : theta1,
        "cost_history"  : cost_history,
        "theta0_history": theta0_history,
        "theta1_history": theta1_history,
        "iterations"    : i + 1,
        "converged"     : converged,
    }


# ---------------------------------------------------------------------------
# 4. Función principal: fit
# ---------------------------------------------------------------------------

def fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.01,
    max_iter: int = 10_000,
    epsilon: float = 1e-6,
    theta0_init: float = 0.0,
    theta1_init: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Función principal: ajusta un modelo de regresión lineal a los datos (X, y).

    Esta función integra todas las piezas: inicializa los parámetros, ejecuta
    el gradiente descendente y devuelve los resultados del modelo junto con
    metadatos de entrenamiento.

    Parámetros
    ----------
    X : np.ndarray
        Array 1-D con los valores de la variable predictora (features).
    y : np.ndarray
        Array 1-D con los valores objetivo (target).
    alpha : float, opcional
        Tasa de aprendizaje. Por defecto 0.01.
    max_iter : int, opcional
        Número máximo de iteraciones del gradiente descendente. Por defecto 10 000.
    epsilon : float, opcional
        Umbral de convergencia. Por defecto 1e-6.
    theta0_init : float, opcional
        Valor inicial de θ₀. Por defecto 0.0.
    theta1_init : float, opcional
        Valor inicial de θ₁. Por defecto 0.0.
    verbose : bool, opcional
        Si es True, imprime un resumen del entrenamiento. Por defecto True.

    Retorna
    -------
    dict con las siguientes claves:
        theta0        : float  — Intercepto óptimo θ₀.
        theta1        : float  — Pendiente óptima θ₁.
        cost_final    : float  — Valor final de la función de coste J.
        cost_history  : list   — Historial de J por iteración.
        theta0_history: list   — Historial de θ₀ por iteración.
        theta1_history: list   — Historial de θ₁ por iteración.
        iterations    : int    — Iteraciones ejecutadas.
        converged     : bool   — Si el algoritmo convergió.

    Ejemplo
    -------
    >>> import numpy as np
    >>> from linreg_gd import fit
    >>> X = np.linspace(0, 1, 100)
    >>> y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)
    >>> modelo = fit(X, y, alpha=0.1, max_iter=10000)
    >>> print(f"θ₀ = {modelo['theta0']:.4f}, θ₁ = {modelo['theta1']:.4f}")
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim != 1 or y.ndim != 1:
        raise ValueError("X e y deben ser arrays 1-D.")
    if len(X) != len(y):
        raise ValueError(
            f"X e y deben tener el mismo tamaño. "
            f"Recibidos: len(X)={len(X)}, len(y)={len(y)}"
        )

    resultado = gradient_descent(
        X, y,
        theta0_init=theta0_init,
        theta1_init=theta1_init,
        alpha=alpha,
        max_iter=max_iter,
        epsilon=epsilon,
    )

    costo_final = cost_function(resultado["theta0"], resultado["theta1"], X, y)

    if verbose:
        estado = "Convergió ✓" if resultado["converged"] else "Alcanzó max_iter"
        print("=" * 45)
        print("  linreg_gd — Resultado del entrenamiento")
        print("=" * 45)
        print(f"  θ₀ (intercepto) : {resultado['theta0']:.6f}")
        print(f"  θ₁ (pendiente)  : {resultado['theta1']:.6f}")
        print(f"  Costo final J   : {costo_final:.8f}")
        print(f"  Iteraciones     : {resultado['iterations']}")
        print(f"  Estado          : {estado}")
        print("=" * 45)

    return {
        "theta0"        : resultado["theta0"],
        "theta1"        : resultado["theta1"],
        "cost_final"    : costo_final,
        "cost_history"  : resultado["cost_history"],
        "theta0_history": resultado["theta0_history"],
        "theta1_history": resultado["theta1_history"],
        "iterations"    : resultado["iterations"],
        "converged"     : resultado["converged"],
    }