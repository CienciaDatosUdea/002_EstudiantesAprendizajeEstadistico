# fcii_regression

Librería de Regresión Lineal con Gradiente Descendente desarrollada para el Laboratorio 4 del curso **Fundamentos de Computación II** de la Universidad de Antioquia.

## Descripción

Esta librería implementa:

- **Gradiente Descendente**: Algoritmo de optimización iterativo que minimiza funciones usando aproximación numérica del gradiente (diferencias finitas).
- **Regresión Lineal Simple**: Modelo de la forma `y = θ₀ + θ₁ * x` entrenado minimizando el Error Cuadrático Medio (MSE).

## Instalación

### Desde el directorio local (desarrollo)

```bash
cd fcii_regression
pip install -e .
```

### Verificar instalación

```python
from fcii_regression import GradientDescent, LinearRegression
print("Instalación exitosa!")
```

## Uso Rápido

### Ejemplo básico de Regresión Lineal

```python
import numpy as np
from fcii_regression import GradientDescent, LinearRegression

# 1. Generar datos sintéticos
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2.5 * X + 1.0 + np.random.randn(50) * 2  # y = 2.5x + 1 + ruido

# 2. Configurar el optimizador
gd = GradientDescent(
    n_iterations=1000,    # Número de iteraciones
    learning_rate=0.01,   # Tasa de aprendizaje
    h=0.001,              # Paso para diferencias finitas
    verbose=False         # No imprimir progreso
)

# 3. Crear y entrenar el modelo
model = LinearRegression(optimizer=gd)
model.fit(X, y)

# 4. Ver resultados
print(model.get_equation())      # y = 1.0234 + 2.4876 * x
print(f"R² Score: {model.score(X, y):.4f}")

# 5. Hacer predicciones
y_pred = model.predict(X)
```

### Visualizaciones

```python
import matplotlib.pyplot as plt

# Gráfica del ajuste
model.plot_fit(X, y, title="Regresión Lineal - Ajuste")
plt.show()

# Evolución del coste durante entrenamiento
model.plot_cost_history()
plt.show()

# Superficie 3D de la función de coste
model.plot_cost_surface(theta0_range=(-2, 4), theta1_range=(0, 5))
plt.show()

# Mapa de contorno de la función de coste
model.plot_cost_contour(theta0_range=(-2, 4), theta1_range=(0, 5))
plt.show()
```

## API Reference

### Clase `GradientDescent`

Optimizador de gradiente descendente genérico.

```python
GradientDescent(
    n_iterations=1000,   # Número máximo de iteraciones
    learning_rate=0.01,  # Tasa de aprendizaje (α)
    h=0.001,             # Paso para aproximación numérica
    verbose=False        # Imprimir progreso cada 100 iteraciones
)
```

**Métodos:**
- `minimize(func, initial_point=None)`: Encuentra el mínimo de una función.
- `plot_3d_surface(func, ...)`: Grafica superficie 3D con trayectoria.
- `plot_contour(func, ...)`: Grafica mapa de contorno con trayectoria.
- `plot_convergence()`: Grafica evolución del valor de la función.

### Clase `LinearRegression`

Modelo de regresión lineal simple.

```python
LinearRegression(optimizer=None)  # Si None, usa GradientDescent por defecto
```

**Métodos:**
- `fit(X, y)`: Entrena el modelo.
- `predict(X)`: Realiza predicciones.
- `score(X, y)`: Calcula el coeficiente R².
- `get_equation()`: Devuelve la ecuación como string.
- `plot_fit(X, y)`: Grafica datos y línea de regresión.
- `plot_cost_history()`: Grafica evolución del MSE.
- `plot_cost_surface(...)`: Grafica superficie 3D del MSE.
- `plot_cost_contour(...)`: Grafica contorno del MSE.

**Atributos (después de `fit`):**
- `theta_`: Array [θ₀, θ₁] con los parámetros óptimos.
- `cost_history_`: Array con valores del MSE en cada iteración.

### Funciones de Coste

```python
from fcii_regression import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
```

## Estructura del Proyecto

```
fcii_regression/
├── fcii_regression/
│   ├── __init__.py          # Exports públicos
│   ├── gradient_descent.py  # Clase GradientDescent
│   ├── linear_regression.py # Clase LinearRegression
│   └── cost_functions.py    # Funciones MSE
├── setup.py                 # Configuración pip
├── README.md                # Este archivo
└── example.py               # Ejemplo completo
```

## Requisitos

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.4.0

## Teoría

### Gradiente Descendente

El gradiente descendente es un algoritmo de optimización que busca el mínimo de una función moviéndose iterativamente en la dirección opuesta al gradiente:

```
θ_new = θ_old - α * ∇J(θ)
```

Donde:
- `α` es el learning rate (controla el tamaño del paso)
- `∇J(θ)` es el gradiente de la función de coste

### Aproximación Numérica del Gradiente

Cuando la derivada analítica es compleja, usamos **diferencias finitas**:

```
∂J/∂θᵢ ≈ (J(θ + h*eᵢ) - J(θ)) / h
```

### Función de Coste MSE

El Error Cuadrático Medio mide qué tan lejos están las predicciones de los valores reales:

```
J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)²
```

## Licencia

MIT License - Uso educativo para el curso FCII de la Universidad de Antioquia.
