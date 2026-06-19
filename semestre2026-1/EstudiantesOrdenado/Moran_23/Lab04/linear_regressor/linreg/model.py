# Contenido de linreg/model.py
import numpy as np
 
def hipotesis(X, theta0, theta1):
    """
    Calcula la prediccion del modelo lineal.
    h(X) = theta0 + theta1 * X
    """
    return theta0 + theta1 * X
 
def costo(X, y, theta0, theta1):
    """
    Calcula la funcion de coste cuadratica (MSE/2).
    J = (1/2m) * sum((h(x_i) - y_i)^2)
    """
    m = len(y)
    return (1/(2*m)) * np.sum((hipotesis(X, theta0, theta1) - y)**2)
 
def gradiente_descendente(X, y, alpha=0.1, epsilon=1e-6, max_iter=100000):
    """
    Ejecuta el algoritmo de gradiente descendente.
    Retorna: (theta0, theta1, historial_J)
    """
    theta0, theta1 = 0.0, 0.0
    m = len(y)
    hist = []
    for i in range(max_iter):
        error = hipotesis(X, theta0, theta1) - y
        theta0 -= alpha * (1/m) * np.sum(error)
        theta1 -= alpha * (1/m) * np.sum(error * X)
        J = costo(X, y, theta0, theta1)
        hist.append(J)
        if i > 0 and abs(hist[-2] - J) < epsilon:
            break
    return theta0, theta1, hist
 
def ajustar(X, y, alpha=0.1, epsilon=1e-6, max_iter=100000):
    """
    Funcion principal: ajusta una regresion lineal a los datos.
    Retorna dict con theta0, theta1, historial_J.
    """
    t0, t1, hist = gradiente_descendente(X, y, alpha, epsilon, max_iter)
    return {'theta0': t0, 'theta1': t1, 'historial_J': hist}
