import numpy as np

def hipotesis(X: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """
    Calcula la hipótesis lineal y_hat = theta0 + theta1 * X.

    Parametros:
        X (np.ndarray): Datos de entrada (variable independiente).
        theta0 (float): Parámetro de ordenadas al origen (intercepto).
        theta1 (float): Parámetro de pendiente.

    Retorna:
        np.ndarray: Valores predichos por el modelo lineal.
    """
    return theta0 + theta1 * np.asarray(X, dtype=np.float64)


def cost_function(X: np.ndarray, y: np.ndarray, theta0: float, theta1: float) -> float:
    """
    Calcula la función de coste cuadrática (Mean Squared Error / 2).

    Parametros:
        X (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente (valores reales).
        theta0 (float): Intercepto.
        theta1 (float): Pendiente.

    Retorna:
        float: Valor del coste J(theta0, theta1).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = len(y)
    
    predictions = hipotesis(X, theta0, theta1)
    cost = (1.0 / (2.0 * m)) * np.sum((predictions - y) ** 2)
    return float(cost)


def gradient_descent(X: np.ndarray, y: np.ndarray, 
                     theta0_init: float = 0.0, theta1_init: float = 0.0, 
                     alpha: float = 0.01, epsilon: float = 1e-14, 
                     max_iter: int = 100000):
    """
    Ejecuta el algoritmo del Descenso de Gradiente para optimizar theta0 y theta1.

    Parametros:
        X (np.ndarray): Variable independiente.
        y (np.ndarray): Variable dependiente.
        theta0_init (float): Valor inicial para theta0.
        theta1_init (float): Valor inicial para theta1.
        alpha (float): Tasa de aprendizaje (learning rate).
        epsilon (float): Criterio de parada basado en la magnitud del gradiente.
        max_iter (int): Número máximo de iteraciones.

    Retorna:
        tuple: (theta0_optimo, theta1_optimo, historial_de_coste)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = len(y)

    theta0 = np.float64(theta0_init)
    theta1 = np.float64(theta1_init)
    cost_history = []

    for i in range(max_iter):
        predictions = hipotesis(X, theta0, theta1)
        
        # Derivadas parciales exactas del MSE
        d_theta0 = (1.0 / m) * np.sum(predictions - y)
        d_theta1 = (1.0 / m) * np.sum((predictions - y) * X)

        # Norma Euclídea del gradiente
        grad_norm = np.sqrt(d_theta0**2 + d_theta1**2)

        # Registrar el coste actual
        current_cost = cost_function(X, y, theta0, theta1)
        cost_history.append(current_cost)

        # Criterio de convergencia
        if grad_norm < epsilon:
            break

        # Actualización simultánea de parámetros
        theta0 -= alpha * d_theta0
        theta1 -= alpha * d_theta1

    return float(theta0), float(theta1), cost_history


def fit_linear_regression(X: np.ndarray, y: np.ndarray, 
                          alpha: float = 0.01, epsilon: float = 1e-14, 
                          max_iter: int = 200000):
    """
    Función principal para ajustar un modelo de regresión lineal sobre los datos.

    Parametros:
        X (np.ndarray): Vector/lista de características.
        y (np.ndarray): Vector/lista de etiquetas objetivo.
        alpha (float): Tasa de aprendizaje.
        epsilon (float): Tolerancia del gradiente.
        max_iter (int): Iteraciones máximas.

    Retorna:
        tuple: (theta0_opt, theta1_opt, historial_coste)
    """
    return gradient_descent(X, y, theta0_init=0.0, theta1_init=0.0, 
                            alpha=alpha, epsilon=epsilon, max_iter=max_iter)


def predict(X: np.ndarray, theta0: float, theta1: float) -> np.ndarray:
    """
    Realiza predicciones sobre un dataset X usando los parámetros theta0 y theta1.
    """
    return hipotesis(X, theta0, theta1)