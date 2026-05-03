"""
Ejemplo de uso de la librería linreg_gd.

Ejecutar desde la raíz del paquete:
    pip install -e .
    python example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from linreg_gd import fit, hypothesis, cost_function

# ─── Generar datos ───
np.random.seed(42)
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

# ─── Ajustar modelo ───
result = fit(X, y, alpha=0.5)

print("=" * 50)
print("  Regresión Lineal con Gradiente Descendente")
print("=" * 50)
print(f"  θ₀ (intercepto): {result['theta0']:.6f}")
print(f"  θ₁ (pendiente):  {result['theta1']:.6f}")
print(f"  Iteraciones:      {result['n_iterations']}")
print(f"  Coste final:      {result['final_cost']:.8f}")
print("=" * 50)

# ─── Visualizar ───
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Ajuste
axes[0].scatter(X, y, alpha=0.5, s=20, label="Datos")
axes[0].plot(
    X,
    hypothesis(X, result["theta0"], result["theta1"]),
    "r-",
    linewidth=2,
    label=f"Ajuste: {result['theta0']:.3f} + {result['theta1']:.3f}X",
)
axes[0].set_xlabel("X")
axes[0].set_ylabel("y")
axes[0].set_title("Regresión Lineal (Gradiente Descendente)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Convergencia
axes[1].plot(result["cost_history"], "b-")
axes[1].set_xlabel("Iteración")
axes[1].set_ylabel("J(θ₀, θ₁)")
axes[1].set_title("Convergencia del coste")
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ejemplo_resultado.png", dpi=100)
plt.show()
print("\nGráfica guardada en ejemplo_resultado.png")
