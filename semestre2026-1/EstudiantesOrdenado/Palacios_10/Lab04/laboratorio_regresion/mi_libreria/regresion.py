import numpy as np

def calcular_hipotesis(X, theta0, theta1):
    """
    Calcula la hipótesis lineal: h(x) = theta0 + theta1 * X
    """
    return theta0 + theta1 * X

def calcular_coste(X, y, theta0, theta1):
    """
    Calcula el Error Cuadrático Medio (Función de coste J).
    """
    m = len(y)
    h = calcular_hipotesis(X, theta0, theta1)
    error = h - y
    return (1 / (2 * m)) * np.sum(error**2)

def ejecutar_gradiente(X, y, theta0, theta1, alpha):
    """
    Ejecuta un paso del gradiente descendente para actualizar theta0 y theta1.
    """
    m = len(y)
    h = calcular_hipotesis(X, theta0, theta1)
    error = h - y
    
    # Derivadas parciales
    d_theta0 = (1 / m) * np.sum(error)
    d_theta1 = (1 / m) * np.sum(error * X)
    
    # Actualización
    nuevo_theta0 = theta0 - alpha * d_theta0
    nuevo_theta1 = theta1 - alpha * d_theta1
    
    return nuevo_theta0, nuevo_theta1

def ajustar_modelo(X, y, alpha=0.01, iteraciones=1000):
    """
    Función principal que entrena el modelo de regresión lineal.
    Retorna los parámetros finales y el historial de costo.
    """
    theta0, theta1 = 0.0, 0.0
    historial_costo = []
    
    for i in range(iteraciones):
        theta0, theta1 = ejecutar_gradiente(X, y, theta0, theta1, alpha)
        costo = calcular_coste(X, y, theta0, theta1)
        historial_costo.append(costo)
        
    return theta0, theta1, historial_costo