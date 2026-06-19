import numpy as np
import matplotlib.pyplot as plt
# Importamos desde la nueva librería instalada
from mi_libreria.regresion import ajustar_modelo, calcular_hipotesis

# 1. Datos del laboratorio
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

# 2. Uso de la librería
t0, t1, costo_hist = ajustar_modelo(X, y, alpha=0.1, iteraciones=1000)

# 3. Resultados
print(f"Modelo ajustado: h(x) = {t0:.4f} + {t1:.4f} * X")

plt.scatter(X, y, label="Datos Reales")
plt.title("Regresión Lineal con mi Librería")
plt.plot(X, calcular_hipotesis(X, t0, t1), color='red', label="Predicción Librería")
plt.legend()
plt.show()