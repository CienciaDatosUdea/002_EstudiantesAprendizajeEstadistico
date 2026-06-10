"""
example.py
----------
Ejemplo de uso de la librería linreg_gd con los datos del Laboratorio 4.

Instalación previa (desde la carpeta Laboratorio_4/):
    pip install .

Ejecución:
    python example.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Importar la librería instalada ──────────────────────────────────────────
from linreg_gd import fit, hypothesis, cost_function

# ── 1. Generar los datos del laboratorio ────────────────────────────────────
np.random.seed(42)  # Fijamos semilla para reproducibilidad
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

print("=" * 50)
print("  Ejemplo — linreg_gd (Laboratorio 4)")
print("=" * 50)
print(f"  Muestras de entrenamiento : {len(X)}")
print(f"  Rango de X                : [{X.min():.2f}, {X.max():.2f}]")
print(f"  Rango de y                : [{y.min():.4f}, {y.max():.4f}]")
print()

# ── 2. Ajustar el modelo con fit() ──────────────────────────────────────────
modelo = fit(
    X,
    y,
    alpha     = 0.1,      # Tasa de aprendizaje
    max_iter  = 10_000,   # Iteraciones máximas
    epsilon   = 1e-6,     # Umbral de convergencia
    verbose   = True,
)

theta0 = modelo["theta0"]
theta1 = modelo["theta1"]

# ── 3. Predicciones ─────────────────────────────────────────────────────────
y_pred = hypothesis(theta0, theta1, X)

# ── 4. Comparación con valores reales de la ley generadora ──────────────────
print()
print("  Valores esperados (ley generadora):")
print(f"    θ₀ real ≈ 0.2000   →  θ₀ estimado = {theta0:.4f}")
print(f"    θ₁ real ≈ 0.2000   →  θ₁ estimado = {theta1:.4f}")
print()

# ── 5. Gráficas ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("linreg_gd — Resultados del Laboratorio 4", fontsize=13)

# --- 5.1 Ajuste del modelo ---
ax1 = axes[0]
ax1.scatter(X, y, s=12, alpha=0.6, label="Datos de entrenamiento")
ax1.plot(X, y_pred, color="crimson", linewidth=2,
         label=f"Modelo: ŷ = {theta0:.3f} + {theta1:.3f}·X")
ax1.set_title("Ajuste del modelo")
ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- 5.2 Convergencia de la función de coste ---
ax2 = axes[1]
ax2.plot(modelo["cost_history"], color="steelblue", linewidth=1.5)
ax2.set_title("Convergencia de J(θ)")
ax2.set_xlabel("Iteración")
ax2.set_ylabel("J(θ₀, θ₁)")
ax2.grid(True, alpha=0.3)
ax2.set_yscale("log")   # Escala log para ver la caída completa

# --- 5.3 Trayectoria en el espacio de parámetros ---
ax3 = axes[2]
t0_hist = np.array(modelo["theta0_history"])
t1_hist = np.array(modelo["theta1_history"])
ax3.plot(t0_hist, t1_hist, color="gray", linewidth=0.8, alpha=0.5)
ax3.scatter(t0_hist[0],  t1_hist[0],  color="green",  s=60, zorder=5,
            label=f"Inicio ({t0_hist[0]:.2f}, {t1_hist[0]:.2f})")
ax3.scatter(theta0, theta1, color="crimson", s=60, marker="X", zorder=5,
            label=f"Mínimo ({theta0:.3f}, {theta1:.3f})")
ax3.set_title("Trayectoria del gradiente descendente")
ax3.set_xlabel("θ₀")
ax3.set_ylabel("θ₁")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("resultado_linreg_gd.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Gráfica guardada como: resultado_linreg_gd.png")

# ── 6. Uso individual de cada función de la librería ────────────────────────
print()
print("── Demostración de funciones individuales ──────────────────────")

X_prueba = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
y_prueba = hypothesis(theta0, theta1, X_prueba)
print(f"\nhypothesis() — predicciones para X = {X_prueba}:")
for xi, yi in zip(X_prueba, y_prueba):
    print(f"  h({xi:.2f}) = {yi:.4f}")

J = cost_function(theta0, theta1, X, y)
print(f"\ncost_function() — coste final con los parámetros óptimos:")
print(f"  J({theta0:.4f}, {theta1:.4f}) = {J:.8f}")