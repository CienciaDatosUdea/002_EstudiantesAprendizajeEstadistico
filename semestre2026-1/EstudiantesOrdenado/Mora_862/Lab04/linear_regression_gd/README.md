# Linear Regression with Gradient Descent

Una librería para implementar regresión lineal univariada usando el algoritmo de gradiente descendente como método de optimización.


## Instalación

### Instalación desde el directorio local

```bash
# Navega al directorio del paquete
cd linear_regression_gd

# Instala en modo normal
pip install .

# O en modo desarrollo
pip install -e .
```

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Dependencias

- **numpy** >= 1.21.0: Para operaciones con arrays y matrices
- **matplotlib** >= 3.4.0: Para visualización de gráficas

## Uso Rápido

### Ejemplo

```python
import numpy as np
from linear_regression_gd import LinearRegression

# Generar datos de ejemplo
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Crear instancia del modelo
model = LinearRegression()

# Entrenar el modelo
result = model.fit(X, y, learning_rate=0.01, verbose=True)

# Hacer predicciones
y_pred = model.predict(X)

# Evaluar el modelo
r2 = model.score(X, y)
print(f"R² = {r2:.4f}")

# Visualizar resultados
model.plot_results(X, y)

# Información del modelo
print(model)
# Output: LinearRegression(theta_0=0.500000, theta_1=1.000000)
```



### Uso de Funciones de Utilidad

### Clase `LinearRegression`

#### Atributos

- `theta_0` (float): Parámetro de intercepto del modelo
- `theta_1` (float): Parámetro de pendiente del modelo
- `cost_history` (list): Historial de valores de coste durante el entrenamiento
- `params_history` (list): Historial de parámetros durante el entrenamiento

#### Métodos

##### `__init__()`
Inicializa una instancia del modelo.

##### `hypothesis(X: np.ndarray) -> np.ndarray`
Calcula la hipótesis lineal: $h(X) = \theta_0 + \theta_1 \cdot X$

**Parámetros:**
- `X`: Array con variables independientes

**Retorna:**
- Array con predicciones

##### `cost_function(X: np.ndarray, y: np.ndarray) -> float`
Calcula la función de coste (Error Cuadrado Medio):

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$

**Parámetros:**
- `X`: Variables independientes
- `y`: Variables dependientes observadas

**Retorna:**
- float: Valor de la función de coste

##### `gradient_descent(X, y, learning_rate=0.01, epsilon=1e-6, max_iterations=10000, verbose=False) -> dict`
Ejecuta el algoritmo de gradiente descendente.

**Parámetros:**
- `X`: Variables independientes
- `y`: Variables dependientes
- `learning_rate` (float): Tasa de aprendizaje α (por defecto: 0.01)
- `epsilon` (float): Criterio de convergencia (por defecto: 1e-6)
- `max_iterations` (int): Máximo de iteraciones (por defecto: 10000)
- `verbose` (bool): Imprimir información de progreso

**Retorna:**
- dict con claves: `theta_0`, `theta_1`, `cost`, `iterations`, `converged`

##### `fit(X, y, learning_rate=0.01, epsilon=1e-6, max_iterations=10000, verbose=False) -> LinearRegression`
Método principal para entrenar el modelo (alias para `gradient_descent`).

**Parámetros:** Iguales a `gradient_descent`

**Retorna:** El modelo (self), permitiendo encadenamiento

##### `predict(X: np.ndarray) -> np.ndarray`
Realiza predicciones sobre nuevos datos.

**Parámetros:**
- `X`: Array con variables independientes

**Retorna:**
- Array con predicciones

##### `score(X: np.ndarray, y: np.ndarray) -> float`
Calcula el coeficiente de determinación $R^2$.

$$R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$$

**Parámetros:**
- `X`: Variables independientes
- `y`: Variables dependientes observadas

**Retorna:**
- float: Coeficiente R² (entre 0 y 1)

##### `plot_results(X, y, title=None)`
Genera gráficas del ajuste y convergencia.

**Parámetros:**
- `X`: Variables independientes
- `y`: Variables dependientes
- `title` (str): Título opcional

### Funciones de Utilidad

#### `linear_hypothesis(X, theta_0, theta_1) -> np.ndarray`
Calcula la hipótesis lineal sin instanciar la clase.

#### `compute_cost(X, y, theta_0, theta_1) -> float`
Calcula la función de coste sin instanciar la clase.

## Teoría: Gradiente Descendente

El algoritmo de gradiente descendente minimiza la función de coste iterativamente:

$$\theta_0 := \theta_0 - \alpha \frac{\partial J}{\partial \theta_0}$$

$$\theta_1 := \theta_1 - \alpha \frac{\partial J}{\partial \theta_1}$$

donde $\alpha$ es la tasa de aprendizaje y las derivadas parciales son:

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$


## Solución de Problemas

### El modelo no converge
- Aumenta la tasa de aprendizaje (`learning_rate`)
- Aumenta el número de iteraciones (`max_iterations`)
- Normaliza los datos antes de entrenar

### R² es muy bajo
- El modelo lineal puede no ser apropiado para tus datos
- Intenta normalizar o escalar los datos
- Verifica si hay valores atípicos

### Error: "learning_rate debe ser positivo"
- Asegúrate de pasar un valor positivo para `learning_rate`
- Valores típicos: 0.001, 0.01, 0.1, 0.5

## Estructura del Paquete

```
linear_regression_gd/
├── linear_regression_gd/
│   ├── __init__.py          # Punto de entrada del paquete
│   └── regression.py        # Implementación principal
├── examples/
│   └── example_usage.py     # Ejemplos de uso
├── setup.py                 # Script de instalación
├── requirements.txt         # Dependencias
└── README.md                # Este archivo
```


## Referencias

- [How to Create Your Own Python Library: A Step-by-Step Guide](https://medium.com/@parasaroraee how-to-create-your-own-python-library-a-step-by-step-guide-6a4f151006d0)
- [Cómo construir tu primer paquete de Python](https://www.freecodecamp.org/espanol/news/como-construir-tu-primer-paquete-de-python/)
- [Gradient Descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent)


