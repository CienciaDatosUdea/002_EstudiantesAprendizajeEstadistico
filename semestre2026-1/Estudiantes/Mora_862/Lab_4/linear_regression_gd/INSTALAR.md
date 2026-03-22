# RESUMEN: Librería Linear Regression with Gradient Descent

## Ubicación

```
/Users/fernanda/Desktop/linear_regression_gd/
```

## Estructura del Paquete

```
linear_regression_gd/
├── linear_regression_gd/
│   ├── __init__.py              ✓ Punto de entrada del paquete
│   └── regression.py            ✓ Implementación principal (~400 líneas con docstrings)
├── examples/
│   └── example_usage.py         ✓ Script ejecutable con 6 ejemplos diferentes
├── setup.py                     ✓ Instalación con pip
├── setup.cfg                    ✓ Configuración alternativa
├── requirements.txt             ✓ Dependencias (numpy, matplotlib)
├── README.md                    ✓ Documentación completa (200+ líneas)
├── .gitignore                   ✓ Archivos a ignorar para git
└── INSTALAR.md                  ✓ Este archivo
```

## Instalación

### Opción 1: Instalación en modo desarrollo (RECOMENDADO)

```bash
cd /Users/fernanda/Desktop/linear_regression_gd
pip install -e .
```

**Ventajas:** Cambios en el código se reflejan inmediatamente sin reinstalar.

### Opción 2: Instalación normal

```bash
cd /Users/fernanda/Desktop/linear_regression_gd
pip install .
```

### Opción 3: Instalación de dependencias solamente

```bash
cd /Users/fernanda/Desktop/linear_regression_gd
pip install -r requirements.txt
```

## Verificación de Instalación

```bash
python -c "from linear_regression_gd import LinearRegression; print('✓ Librería instalada correctamente')"
```

## Uso Básico

```python
import numpy as np
from linear_regression_gd import LinearRegression

# Crear datos
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Crear modelo
model = LinearRegression()

# Entrenar
model.fit(X, y, learning_rate=0.01)

# Predecir
predictions = model.predict(X)

# Evaluar
r2 = model.score(X, y)
print(f"R² = {r2:.4f}")

# Visualizar
model.plot_results(X, y)
```

## Características de la Librería

### Clase LinearRegression

**Inicialización:**
```python
model = LinearRegression()
```

**Métodos principales:**

1. **fit()** - Entrenamiento
   ```python
   model.fit(X, y, learning_rate=0.01, epsilon=1e-6, max_iterations=10000, verbose=False)
   ```
   - `X`: Array 1D con variables independientes
   - `y`: Array 1D con variables dependientes
   - `learning_rate`: Tasa de aprendizaje (por defecto: 0.01)
   - `epsilon`: Criterio de convergencia (por defecto: 1e-6)
   - `max_iterations`: Número máximo de iteraciones (por defecto: 10000)
   - `verbose`: Imprimir información de progreso (por defecto: False)

2. **predict()** - Predicción
   ```python
   y_pred = model.predict(X)
   ```

3. **score()** - Evaluación (R²)
   ```python
   r2 = model.score(X, y)
   ```

4. **plot_results()** - Visualización
   ```python
   model.plot_results(X, y, title="Mi Gráfica")
   ```

5. **hypothesis()**, **cost_function()** - Funciones individuales

### Atributos del Modelo

- `theta_0`: Parámetro de intercepto (pendiente y)
- `theta_1`: Parámetro de pendiente (gradiente de la recta)
- `cost_history`: Historial de valores de coste durante el entrenamiento
- `params_history`: Historial de parámetros durante el entrenamiento

### Funciones de Utilidad

```python
from linear_regression_gd import linear_hypothesis, compute_cost

# Calcular hipótesis directamente
h = linear_hypothesis(X, theta_0=2.0, theta_1=3.0)

# Calcular coste directamente
J = compute_cost(X, y, theta_0=2.0, theta_1=3.0)
```

## Ejemplo Completo

```python
import numpy as np
from linear_regression_gd import LinearRegression
from sklearn.linear_model import LinearRegression as SKLin
from sklearn.metrics import r2_score

# Datos (con ruido)
np.random.seed(42)
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2*X + 0.02*np.random.random(100)

# Nuestro modelo
model = LinearRegression()
model.fit(X, y, learning_rate=0.01, verbose=True)

print(f"\nNuestra librería:")
print(f"  θ₀ = {model.theta_0:.6f}")
print(f"  θ₁ = {model.theta_1:.6f}")
print(f"  R² = {model.score(X, y):.6f}")

# Comparación con sklearn
sk_model = SKLin()
sk_model.fit(X.reshape(-1, 1), y)
print(f"\nsklearn:")
print(f"  θ₀ = {sk_model.intercept_:.6f}")
print(f"  θ₁ = {sk_model.coef_[0]:.6f}")
print(f"  R² = {r2_score(y, sk_model.predict(X.reshape(-1, 1))):.6f}")

# Visualizar
model.plot_results(X, y)
```

## Ejecutar Ejemplos

### Script Python directo

```bash
cd /Users/fernanda/Desktop/linear_regression_gd
python examples/example_usage.py
```

### Jupyter Notebook


1. Busca la celda "Demostración: Librería linear_regression_gd"
2. Ejecuta la celda

### Interactive Python

```bash
python
>>> from linear_regression_gd import LinearRegression
>>> import numpy as np
>>> X = np.array([1, 2, 3, 4, 5], dtype=float)
>>> y = np.array([2, 4, 5, 4, 5], dtype=float)
>>> model = LinearRegression()
>>> model.fit(X, y)
>>> model.plot_results(X, y)
```

## Parámetros Recomendados

| Parámetro | Rango | Recomendación |
|-----------|-------|---------------|
| learning_rate | 0.001 - 1.0 | Empezar con 0.01 |
| epsilon | 1e-8 - 1e-4 | 1e-6 (defecto) es bueno |
| max_iterations | 1000 - 100000 | 10000 es suficiente |

## Solución de Problemas

### "ModuleNotFoundError: No module named 'linear_regression_gd'"
→ Ejecuta: `pip install -e /Users/fernanda/Desktop/linear_regression_gd`

### El modelo no converge
→ Intenta aumentar `learning_rate` (ej: 0.1) o `epsilon` (ej: 1e-4)

### R² muy bajo
→ Verifica si los datos tienen relación lineal. Prueba normalizar los datos.

### Resultados diferentes a sklearn
→ Es normal. Las pequeñas diferencias (<0.01%) son esperadas debido a métodos internos distintos.

## Documentación

Para documentación detallada de cada función y parámetro:

```bash
python -c "from linear_regression_gd import LinearRegression; help(LinearRegression.fit)"
```

O consulta `README.md` en el directorio de la librería.

## Características Implementadas

-Hipótesis lineal: $h(X) = \theta_0 + \theta_1 X$
-Función de coste: $J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}(h(x^{(i)}) - y^{(i)})^2$
-Gradiente descendente con convergencia personalizable
-Método `fit()` compatible con scikit-learn
-Método `predict()` para predicciones
-Método `score()` para evaluar (R²)
-Visualización de resultados
-Historial de convergencia
-Documentación exhaustiva en docstrings
-Funciones de utilidad para uso directo
-Compatible con Python 3.7+




