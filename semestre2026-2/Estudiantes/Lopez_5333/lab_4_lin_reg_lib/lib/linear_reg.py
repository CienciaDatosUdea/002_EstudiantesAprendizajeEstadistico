import numpy as np

"""
linear_reg.py v0.1

Esta Libreria contiene funciones para realizar regresion lineal usando gradiente descendente
Funciones:
- hipotesis: Calcula la hipótesis para una regresión lineal.
- coste_func_: Calcula el coste cuadrático para una regresión lineal.
- grad_desc_2d: Implementa el gradiente descendente numérico para regresión lineal.
- ajustar_modelo: Función principal para ajustar la regresión lineal a un conjunto de datos.
"""

#------------------------------------------------------------------------------
#funcion de hipotesis para regresion lineal
def hipotesis(Theta, X):
    """
    Calcula la hipótesis para una regresión lineal. (vectorizada)
    theta: array de parámetros [theta_0, theta_1]
    X: array de características
    """
    return np.dot(X, Theta)  # Producto punto punto entre X y theta

#------------------------------------------------------------------------------
#Funcion de coste para regresion lineal (similar a la que usamos anteriormente)
def coste_func_(Theta, X, y):
    """
    Calcula el coste cuadrático para una regresión lineal.
    Theta: array de parámetros [theta_0, theta_1]
    X: array de características
    y: array de valores reales
    """
    m = len(y)
    predict = hipotesis(Theta, X)
    return 1/(2*m) * np.sum((predict - y)**2)

#-----------------------------------------------------------------------------
#Funcion de gradiente descendente para regresion lineal
#Deb ser compatible con la funcion de coste definida anteriormente y la hipotesis
def grad_desc_2d(X, y, Theta, alpha=0.1, h=0.01, max_iter=1000):
    """
    Gradiente descendente numérico para regresión lineal.
    X: matriz de características, incluyendo la columna de unos.
    y: valores reales.
    Theta: parámetros iniciales [theta_0, theta_1].
    alpha: tasa de aprendizaje.
    h: paso para la aproximación numérica del gradiente.
    max_iter: número máximo de iteraciones.
    """

    Theta = np.array(Theta, dtype=float)
    grad = np.zeros_like(Theta)
    trayectoria = [Theta.copy()]

    for _ in range(max_iter):
        for j in range(len(Theta)):
            _T = Theta.copy()
            _T[j] += h

            # Aproximación de la derivada parcial
            grad[j] = (
                coste_func_(_T, X, y) - coste_func_(Theta, X, y)
            ) / h

        # Actualización de los parámetros
        Theta = Theta - alpha * grad
        trayectoria.append(Theta.copy())

    return Theta, np.array(trayectoria)

#-----------------------------------------------------------------------------
#Funcion de ajuste de modelo lineal a un conjunto de datos
def ajustar_modelo(X_train, y_train, alpha=0.1, max_iter=1000):
    """
    Función principal para ajustar la regresión lineal a un conjunto de datos.
    Prepara la matriz de diseño y llama al gradiente descendente.
    
    Args:
        X_train (np.ndarray): Vector de características X (dimensión m).
        y_train (np.ndarray): Vector de etiquetas reales y (dimensión m).
        alpha (float): Tasa de aprendizaje. Default: 0.1
        max_iter (int): Iteraciones máximas. Default: 1000
        
    Returns:
        tuple: (theta_optimo, historial_de_coste)
    """
    m = len(y_train)
    
    # Construcción de la Matriz de Diseño (agregando la columna de 1s para Theta_0)
    # X_b tendrá dimensiones (m, 2)
    X_b = np.c_[np.ones((m, 1)), X_train]
    
    # Inicializar Theta en [0, 0]
    theta_inicial = np.zeros(2)
    
    # Llamar al algoritmo
    theta_optimo, historial = grad_desc_2d(X_b, y_train, theta_inicial, alpha=alpha, max_iter=max_iter)
    
    return theta_optimo, historial