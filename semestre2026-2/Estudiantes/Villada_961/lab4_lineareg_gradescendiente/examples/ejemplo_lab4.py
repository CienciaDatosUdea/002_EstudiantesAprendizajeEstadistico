"""Ejemplo de uso: python examples/ejemplo_lab4.py"""

import numpy as np
from lineareg import ajustar_modelo, hipotesis_lineal


X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.default_rng(17).random(100)
modelo = ajustar_modelo(X, y, tasa_aprendizaje=0.1, tolerancia=1e-12)

print(f"Intercepto: {modelo['intercepto']:.6f}")
print(f"Pendiente: {modelo['pendiente']:.6f}")
print(f"Coste final: {modelo['historial_coste'][-1]:.8f}")
print(f"Prediccion para X=0.5: {hipotesis_lineal(0.5, modelo['theta']):.6f}")
