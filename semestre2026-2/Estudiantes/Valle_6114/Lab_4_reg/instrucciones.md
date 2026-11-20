# Linear Regression GD (Librería de Regresión Lineal)

Librería liviana en Python para ajustar modelos de regresión lineal utilizando el algoritmo de Descenso de Gradiente y la función de coste del Error Cuadrático Medio ($MSE$).

---

##  Estructura del Paquete

```text
linear_regression_gd/
├── pyproject.toml
├── README.md
├── ejemplo.py
└── regresion_lineal/
    ├── __init__.py
    └── core.py
```

---

##  Instalación

Puedes instalar la librería localmente desde el directorio raíz del proyecto ejecutando:

```bash
pip install .
```

Si deseas instalarla en modo desarrollador (para modificar el código sin reinstalar):

```bash
pip install -e .
```

---

##  Guía de Uso Rápido

```python
import numpy as np
from regresion_lineal import fit_linear_regression, predict

# 1. Crear datos de prueba
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2.1, 3.9, 6.1, 8.2, 9.8], dtype=float)

# 2. Ajustar el modelo
theta0, theta1, history = fit_linear_regression(
    X, y, alpha=0.01, epsilon=1e-14, max_iter=100000
)

print(f"Parámetros encontrados: theta0 = {theta0:.4f}, theta1 = {theta1:.4f}")

# 3. Realizar predicciones
y_pred = predict(X, theta0, theta1)
print("Predicciones:", y_pred)
```

---

##  Documentación de Funciones

### `hipotesis(X, theta0, theta1)`
Calcula la predicción lineal $\hat{y} = \theta_0 + \theta_1 X$.
* **`X`** *(np.ndarray)*: Array o lista con las variables independientes.
* **`theta0`** *(float)*: Intercepto del modelo.
* **`theta1`** *(float)*: Pendiente del modelo.
* **Retorna**: Array con los valores predichos $\hat{y}$.

---

### `cost_function(X, y, theta0, theta1)`
Evalúa la función de coste cuadrática (Error Cuadrático Medio):
$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$
* **`X`** *(np.ndarray)*: Entradas del modelo.
* **`y`** *(np.ndarray)*: Valores reales observados.
* **`theta0`, `theta1`** *(float)*: Parámetros del modelo.
* **Retorna**: Valor flotante con el coste $J(\theta_0, \theta_1)$.

---

### `gradient_descent(X, y, theta0_init=0.0, theta1_init=0.0, alpha=0.01, epsilon=1e-14, max_iter=100000)`
Ejecuta el algoritmo iterativo del descenso de gradiente hasta alcanzar la convergencia o el número máximo de iteraciones.
* **`alpha`** *(float)*: Tasa de aprendizaje (learning rate).
* **`epsilon`** *(float)*: Criterio de parada para la norma del gradiente ($10^{-14}$ por defecto).
* **Retorna**: Tupla `(theta0, theta1, history_cost)` con los parámetros óptimos y el historial del coste.

---

### `fit_linear_regression(X, y, alpha=0.01, epsilon=1e-14, max_iter=100000)`
**Función Principal.** Ajusta el modelo lineal a un conjunto de datos `X` e `y`. Convierte automáticamente los datos a arreglos de 64 bits (`np.float64`) para asegurar la tolerancia requerida.
* **Retorna**: `(theta0, theta1, history_cost)`

---

### `predict(X, theta0, theta1)`
Aplica el modelo ajustado sobre un nuevo conjunto de datos `X`.