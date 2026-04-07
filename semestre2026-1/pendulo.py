import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros del péndulo
g = 9.81   # gravedad (m/s²)
L = 1.0    # longitud del péndulo (m)
b = 0.1    # coeficiente de amortiguamiento

# Definición del sistema de ecuaciones diferenciales
# θ'' + (b/mL²)θ' + (g/L)sin(θ) = 0
# Estado: u[0] = θ (ángulo), u[1] = θ' (velocidad angular)
def pendulum(t, u):
    theta, omega = u
    dtheta_dt = omega
    domega_dt = -b * omega - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Condiciones iniciales
theta0 = np.pi / 4   # ángulo inicial (45 grados)
omega0 = 0.0         # velocidad angular inicial

u0 = [theta0, omega0]
tspan = (0.0, 20.0)
t_eval = np.linspace(*tspan, 1000)

# Resolver el sistema de ecuaciones diferenciales
sol = solve_ivp(pendulum, tspan, u0, t_eval=t_eval,
                method='RK45', rtol=1e-8, atol=1e-8)

# Graficar los resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Ángulo vs tiempo
axes[0, 0].plot(sol.t, sol.y[0], color='blue', linewidth=2, label='θ(t)')
axes[0, 0].set_xlabel('Tiempo (s)')
axes[0, 0].set_ylabel('Ángulo θ (rad)')
axes[0, 0].set_title('Péndulo Simple Amortiguado')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Velocidad angular vs tiempo
axes[0, 1].plot(sol.t, sol.y[1], color='red', linewidth=2, label='ω(t)')
axes[0, 1].set_xlabel('Tiempo (s)')
axes[0, 1].set_ylabel('Velocidad Angular ω (rad/s)')
axes[0, 1].set_title('Velocidad Angular')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Espacio de fase
axes[1, 0].plot(sol.y[0], sol.y[1], color='green', linewidth=2, label='Trayectoria')
axes[1, 0].set_xlabel('θ (rad)')
axes[1, 0].set_ylabel('ω (rad/s)')
axes[1, 0].set_title('Espacio de Fase')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Ocultar el subplot vacío
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('pendulo_simulacion.png', dpi=150)
plt.show()
print("Simulación completada. Gráfica guardada en 'pendulo_simulacion.png'")
