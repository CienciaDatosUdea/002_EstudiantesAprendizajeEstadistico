import numpy as np
import matplotlib.pyplot as plt

def hipotesis(X, theta_0, theta_1):
    """Calcula la hipótesis lineal h(X) = theta_0 + theta_1*X"""
    return theta_0 + theta_1 * X

def funcion_coste(X, y, theta_0, theta_1):
    """Calcula el Error Cuadrático Medio."""
    m = len(y)
    return (1 / (2 * m)) * np.sum((hipotesis(X, theta_0, theta_1) - y) ** 2)

def gradiente_descendente(X, y, alpha=0.1, iteraciones=1000):
    """Ejecuta el gradiente descendente iterativamente."""
    m = len(y)
    theta_0, theta_1 = 0.0, 0.0
    for _ in range(iteraciones):
        error = hipotesis(X, theta_0, theta_1) - y
        theta_0 -= alpha * (1/m) * np.sum(error)
        theta_1 -= alpha * (1/m) * np.sum(error * X)
    return theta_0, theta_1

def generar_graficos(X, y, theta_0, theta_1):
    """Genera las gráficas de superficie 3D, contorno y ajuste lineal."""
    # Crear malla para la función de coste
    theta_0_malla = np.linspace(-5, 5, 60)
    theta_1_malla = np.linspace(0, 4, 60)
    T0, T1 = np.meshgrid(theta_0_malla, theta_1_malla)
    J_malla = np.zeros_like(T0)

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            J_malla[i, j] = funcion_coste(X, y, T0[i, j], T1[i, j])

    fig = plt.figure(figsize=(18, 5))

    # Gráfico 1: Superficie 3D
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(T0, T1, J_malla, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_title('Superficie de la Función de Coste')
    ax1.set_xlabel('$\\theta_0$')
    ax1.set_ylabel('$\\theta_1$')
    ax1.set_zlabel('Coste $J$')

    # Gráfico 2: Curvas de Nivel
    ax2 = fig.add_subplot(132)
    ax2.contour(T0, T1, J_malla, levels=np.logspace(-2, 3, 25), cmap='viridis')
    ax2.plot(theta_0, theta_1, 'ro', markersize=8, label=f'Mínimo óptimo')
    ax2.set_title('Curvas de Nivel (Contorno)')
    ax2.set_xlabel('$\\theta_0$')
    ax2.set_ylabel('$\\theta_1$')
    ax2.legend()

    # Gráfico 3: Ajuste lineal sobre los datos
    ax3 = fig.add_subplot(133)
    ax3.scatter(X, y, color='gray', label='Datos')
    ax3.plot(X, hipotesis(X, theta_0, theta_1), 'r--', linewidth=2, label='Ajuste lineal')
    ax3.set_title('Datos vs Recta Ajustada')
    ax3.set_xlabel('X')
    ax3.set_ylabel('y')
    ax3.legend()

    plt.tight_layout()
    plt.show()

def ajustar_modelo(X, y, alpha=0.1, iteraciones=1000, graficar=True):
    """Función principal que envuelve el ajuste del modelo y visualiza los datos."""
    t0, t1 = gradiente_descendente(X, y, alpha, iteraciones)
    coste_final = funcion_coste(X, y, t0, t1)
    
    if graficar:
        generar_graficos(X, y, t0, t1)
        
    return {"theta_0": t0, "theta_1": t1, "coste_final": coste_final}
