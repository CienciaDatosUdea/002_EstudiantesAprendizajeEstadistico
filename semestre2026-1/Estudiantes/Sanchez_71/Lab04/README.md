# Librería de Regresión Lineal (Gradiente Descendente) - Lab 4

Esta librería proporciona herramientas modulares para realizar ajustes de regresión lineal simple utilizando el algoritmo de optimización de Gradiente Descendente. Fue desarrollada como parte de los requisitos del Laboratorio 4.

## 🚀 Instalación

1. Descargue la carpeta `linear_gd_lib` en su directorio de proyecto.
2. Asegúrese de tener instalado `numpy` y `matplotlib`.
3. Importe las funciones necesarias en su script de Python o Jupyter Notebook.

---

## 📚 Documentación Técnica (API)

A continuación se describen las funciones principales incluidas en el módulo `regresion.py`:

### 1. `calcular_hipotesis(X, theta_0, theta_1)`
Calcula los valores predichos ($h_\theta$) para un conjunto de datos dado.
*   **X**: Vector de datos de entrada.
*   **theta_0**: Parámetro de intercepto (sesgo).
*   **theta_1**: Parámetro de pendiente (peso).
*   **Retorna**: Un arreglo con las predicciones.

### 2. `calcular_coste(X, y, theta_0, theta_1)`
Calcula el Error Cuadrático Medio (MSE) para evaluar el desempeño del modelo.
*   **X, y**: Datos de entrada y etiquetas reales.
*   **Retorna**: El valor escalar del coste $J$.

### 3. `ejecutar_gradiente_descendente(X, y, alpha, iteraciones)`
Motor de optimización que actualiza los parámetros iterativamente.
*   **alpha**: Tasa de aprendizaje.
*   **iteraciones**: Número de épocas de entrenamiento.
*   **Retorna**: (theta_0, theta_1, historial_de_costos).

### 4. `ajustar_modelo(X, y, alpha, iteraciones)`
Función principal de alto nivel para el usuario final.
*   **Retorna**: Un diccionario con los parámetros finales y el historial de entrenamiento.

---

## 💻 Ejemplo de Uso

```python
import numpy as np
from linear_gd_lib.regresion import ajustar_modelo

# Datos de ejemplo
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Entrenamiento
resultado = ajustar_modelo(X, y, alpha=0.1, iteraciones=1000)

print(f"Pendiente encontrada: {resultado['theta_1']}")
```

---

## 🤖 Créditos
Este laboratorio y su correspondiente librería fueron desarrollados con la asistencia de Inteligencia Artificial (Google AI Studio / Gemini).


