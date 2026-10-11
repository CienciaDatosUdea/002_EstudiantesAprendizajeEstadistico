# Libreria linear_reg: Regresión Lineal y Gradiente Descendente

Este paquete es una pequeña librería de Python desarrollada como parte del Laboratorio N°4. Contiene implementaciones matemáticas vectorizadas para realizar modelos de Regresión Lineal desde cero, utilizando funciones de coste (Error Cuadrático Medio) y el algoritmo de optimización de Gradiente Descendente (método numérico).

## Estructura de la Librería

La librería consta de las siguientes funciones principales:
1. `hipotesis(Theta, X)`: Calcula las predicciones del modelo lineal de forma matricial.
2. `coste_func_(Theta, X, y)`: Calcula el Error Cuadrático Medio.
3. `grad_desc_2d(...)`: Implementa el gradiente descendente numérico iterativo.
4. `ajustar_modelo(...)`: Función principal de alto nivel que automatiza el proceso de entrenamiento.

---

## Instrucciones de Instalación

Para instalar esta librería localmente en tu entorno de Python, sigue estos pasos:

1. Abre tu terminal de comandos (CMD, PowerShell, o Anaconda Prompt).

2. Navega hasta el directorio raíz donde se encuentra este archivo `README.md` y el archivo `setup.py`. Por ejemplo:
   ```bash
   cd lab_4_lin_reg_lib/

3. Ejecuta el comando de instalación `pip install .`

--

#Ejemplo de Uso 

```python
import numpy as np
from linear_reg import ajustar_modelo, hipotesis

# 1. Generar datos de prueba
X_train = np.linspace(0, 1, 100)
y_train = 0.2 + 0.2 * X_train + 0.02 * np.random.random(100)

# 2. Ajustar el modelo usando Gradiente Descendente
theta_optimo, historial = ajustar_modelo(
    X_train, y_train, alpha=0.5, max_iter=1500
)

print(f"Intercepto encontrado: {theta_optimo[0]:.4f}")
print(f"Pendiente encontrada: {theta_optimo[1]:.4f}")

# 3. Realizar predicciones
# (Recuerda agregar la columna de 1s para el intercepto)
X_b = np.c_[np.ones((len(X_train), 1)), X_train]
predicciones = hipotesis(theta_optimo, X_b)
```