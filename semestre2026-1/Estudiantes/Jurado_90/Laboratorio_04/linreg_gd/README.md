# linreg_gd

Librería de regresión lineal simple mediante gradiente descendente.

## Instalación

```bash
# Desde el directorio raíz del paquete:
pip install -e .

# O directamente:
pip install .
```

## Uso básico

```python
import numpy as np
from linreg_gd import fit, hypothesis

# Generar datos
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

# Ajustar modelo
result = fit(X, y, alpha=0.5)

print(f"θ₀ = {result['theta0']:.6f}")
print(f"θ₁ = {result['theta1']:.6f}")
print(f"Iteraciones: {result['n_iterations']}")
print(f"Coste final: {result['final_cost']:.8f}")

# Predecir
X_new = np.array([0.5, 0.75, 1.0])
y_pred = hypothesis(X_new, result['theta0'], result['theta1'])
print(f"Predicciones: {y_pred}")
```

## API

### `hypothesis(X, theta0, theta1)`
Calcula h(X) = θ₀ + θ₁·X.

### `cost_function(X, y, theta0, theta1)`
Calcula J(θ₀,θ₁) = (1/2m)·Σ(h(xⁱ) - yⁱ)².

### `gradient_descent(X, y, theta0, theta1, alpha, epsilon, max_iter)`
Ejecuta el algoritmo de gradiente descendente. Retorna `(theta0, theta1, cost_history, n_iterations)`.

### `fit(X, y, alpha, epsilon, max_iter)`
Función principal que ajusta el modelo y retorna un diccionario con parámetros óptimos y métricas.

## Requisitos

- Python >= 3.8
- NumPy >= 1.20
