import numpy as np

def calcular_hipotesis(X, theta_0, theta_1):
    """
    Calcula la predicción lineal para un conjunto de datos.
    """
    return theta_0 + theta_1 * X

def calcular_coste(X, y, theta_0, theta_1):
    """
    Calcula el error cuadrático medio del modelo.
    """
    m = len(y)
    predicciones = calcular_hipotesis(X, theta_0, theta_1)
    error = predicciones - y
    coste = (1 / (2 * m)) * np.sum(error**2)
    return coste

def ejecutar_gradiente_descendente(X, y, alpha, iteraciones):
    """
    Ejecuta el algoritmo de optimización para encontrar los parámetros óptimos.
    """
    m = len(y)
    theta_0 = 0.0
    theta_1 = 0.0
    historial_coste = []

    for i in range(iteraciones):
        predicciones = calcular_hipotesis(X, theta_0, theta_1)
        error = predicciones - y

        grad_0 = (1/m) * np.sum(error)
        grad_1 = (1/m) * np.sum(error * X)

        theta_0 = theta_0 - alpha * grad_0
        theta_1 = theta_1 - alpha * grad_1

        coste = calcular_coste(X, y, theta_0, theta_1)
        historial_coste.append(coste)

    return theta_0, theta_1, historial_coste

def ajustar_modelo(X, y, alpha=0.01, iteraciones=1000):
    """
    Función principal que envuelve la lógica de ajuste de regresión lineal.
    """
    t0, t1, costos = ejecutar_gradiente_descendente(X, y, alpha, iteraciones)
    return {"theta_0": t0, "theta_1": t1, "historial_coste": costos}
