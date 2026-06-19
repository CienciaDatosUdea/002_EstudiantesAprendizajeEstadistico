import numpy as np

def hipotesis(X, theta_0, theta_1):
    """
    Calcula la hipótesis lineal h(x) = theta_0 + theta_1 * x
    """
    return theta_0 + theta_1 * X

def funcion_costo(X, y, theta_0, theta_1):
    """
    Calcula la función de coste cuadrática para regresión lineal.
    """
    m = len(y)
    h = hipotesis(X, theta_0, theta_1)
    return (1/(2*m)) * np.sum((h - y)**2)

def gradiente_descendente(X, y, alpha=0.1, epsilon=1e-6, max_iter=10000):
    """
    Ejecuta el gradiente descendente para encontrar los parámetros óptimos.
    """
    m = len(y)
    theta_0, theta_1 = 0.0, 0.0
    for i in range(max_iter):
        h = hipotesis(X, theta_0, theta_1)
        d0 = (1/m) * np.sum(h - y)
        d1 = (1/m) * np.sum((h - y) * X)
        theta_0_new = theta_0 - alpha * d0
        theta_1_new = theta_1 - alpha * d1
        if np.abs(theta_0_new - theta_0) < epsilon and np.abs(theta_1_new - theta_1) < epsilon:
            break
        theta_0, theta_1 = theta_0_new, theta_1_new
    return theta_0, theta_1

def ajustar(X, y, metodo='gradiente', **kwargs):
    """
    Ajusta un modelo de regresión lineal a los datos.
    metodo: 'gradiente' o 'minimos'
    kwargs: parámetros para el gradiente descendente
    """
    if metodo == 'gradiente':
        return gradiente_descendente(X, y, **kwargs)
    elif metodo == 'minimos':
        X_mat = np.vstack([np.ones_like(X), X]).T
        theta, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
        return theta[0], theta[1]
    else:
        raise ValueError("Método no reconocido. Usa 'gradiente' o 'minimos'.")