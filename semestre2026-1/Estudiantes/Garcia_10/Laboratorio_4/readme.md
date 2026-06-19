# linreg_gd

Librería de Python para ajustar una **regresión lineal simple** mediante función de coste cuadrática y gradiente descendente.

Desarrollada como entrega del Laboratorio 4 — Aprendizaje Estadístico UdeA 2026-1.

---

## Instalación

### Requisitos previos

- Python 3.8 o superior
- `pip`

### Instalar desde el código fuente

1. Clona o descarga el repositorio y navega a la carpeta `Laboratorio_4`:

```bash
cd 002_EstudiantesAprendizajeEstadistico/semestre2026-1/Estudiantes/Garcia_10/Laboratorio_4
```

2. Instala el paquete con `pip`:

```bash
pip install .
```

> Para desarrollo (los cambios en el código se reflejan sin reinstalar):
> ```bash
> pip install -e .
> ```

3. Verifica la instalación:

```python
import linreg_gd
print(linreg_gd.__version__)   # 1.0.0
```

---

## Estructura del paquete

```
Laboratorio_4/
├── linreg_gd/
│   ├── __init__.py   ← Interfaz pública del paquete
│   └── core.py       ← Implementación de las funciones
├── setup.py          ← Configuración para pip
├── README.md         ← Este archivo
└── example.py        ← Ejemplo completo con datos del laboratorio
```

---

## Referencia de la API

### `hypothesis(theta0, theta1, X)`

Calcula la hipótesis lineal.

```
h(X) = θ₀ + θ₁·X
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `theta0` | `float` | Intercepto θ₀ |
| `theta1` | `float` | Pendiente θ₁ |
| `X` | `np.ndarray` | Valores de la variable predictora |

**Retorna:** `np.ndarray` con las predicciones.

---

### `cost_function(theta0, theta1, X, y)`

Calcula la función de coste cuadrática (MSE escalado).

```
J(θ₀, θ₁) = (1 / 2m) · Σ (h(xᵢ) − yᵢ)²
```

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `theta0` | `float` | Intercepto θ₀ |
| `theta1` | `float` | Pendiente θ₁ |
| `X` | `np.ndarray` | Valores predictores |
| `y` | `np.ndarray` | Valores reales |

**Retorna:** `float` — valor de J.

---

### `gradient_descent(X, y, theta0_init, theta1_init, alpha, max_iter, epsilon)`

Ejecuta el algoritmo de gradiente descendente.

En cada iteración actualiza θ₀ y θ₁ simultáneamente:

```
θ₀ := θ₀ − α · (1/m) · Σ (h(xᵢ) − yᵢ)
θ₁ := θ₁ − α · (1/m) · Σ (h(xᵢ) − yᵢ)·xᵢ
```

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Valores predictores |
| `y` | `np.ndarray` | — | Valores reales |
| `theta0_init` | `float` | `0.0` | Valor inicial de θ₀ |
| `theta1_init` | `float` | `0.0` | Valor inicial de θ₁ |
| `alpha` | `float` | `0.01` | Tasa de aprendizaje |
| `max_iter` | `int` | `10000` | Máximo de iteraciones |
| `epsilon` | `float` | `1e-6` | Umbral de convergencia |

**Retorna:** `dict` con claves `theta0`, `theta1`, `cost_history`, `theta0_history`, `theta1_history`, `iterations`, `converged`.

---

### `fit(X, y, alpha, max_iter, epsilon, theta0_init, theta1_init, verbose)`

**Función principal.** Ajusta el modelo de regresión lineal a los datos.

Combina todas las funciones anteriores en un único punto de entrada.

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `X` | `np.ndarray` | — | Valores predictores |
| `y` | `np.ndarray` | — | Valores objetivo |
| `alpha` | `float` | `0.01` | Tasa de aprendizaje |
| `max_iter` | `int` | `10000` | Máximo de iteraciones |
| `epsilon` | `float` | `1e-6` | Umbral de convergencia |
| `theta0_init` | `float` | `0.0` | Valor inicial de θ₀ |
| `theta1_init` | `float` | `0.0` | Valor inicial de θ₁ |
| `verbose` | `bool` | `True` | Imprime resumen al terminar |

**Retorna:** `dict` con claves `theta0`, `theta1`, `cost_final`, `cost_history`, `theta0_history`, `theta1_history`, `iterations`, `converged`.

---

## Uso rápido

```python
import numpy as np
from linreg_gd import fit, hypothesis

# Datos de ejemplo (datos del Laboratorio 4)
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

# Ajustar el modelo
modelo = fit(X, y, alpha=0.1, max_iter=10000)

# Predecir nuevos valores
X_nuevo = np.array([0.25, 0.50, 0.75])
y_pred  = hypothesis(modelo["theta0"], modelo["theta1"], X_nuevo)
print(y_pred)
```

Ver el archivo `example.py` para un ejemplo completo con visualizaciones.

---

## Dependencias

| Paquete | Versión mínima |
|---------|---------------|
| Python | 3.8 |
| numpy | 1.21 |

---

## Licencia

MIT