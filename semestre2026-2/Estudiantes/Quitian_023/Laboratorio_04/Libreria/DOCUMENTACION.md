# Documentación de Uso - LinReg Model

Esta librería expone una serie de funciones para calcular, optimizar y visualizar regresiones lineales.

## Referencia de Funciones (API)

### 1. `ajustar_modelo(X, y, alpha=0.1, iteraciones=1000, graficar=True)`
Es la función principal de alto nivel. Toma los datos de entrada, ejecuta el gradiente descendente para encontrar los parámetros óptimos, calcula el coste final y (opcionalmente) genera visualizaciones.
*   **Parámetros:**
    *   `X` (array): Vector de datos de la variable independiente (característica).
    *   `y` (array): Vector de datos de la variable dependiente (etiquetas).
    *   `alpha` (float, opcional): Tasa de aprendizaje. Por defecto es `0.1`.
    *   `iteraciones` (int, opcional): Número máximo de pasos para el gradiente. Por defecto es `1000`.
    *   `graficar` (bool, opcional): Si es `True`, despliega una ventana con la superficie 3D, curvas de nivel y el ajuste sobre los datos.
*   **Retorna:** Un diccionario con las claves `theta_0` (intercepto), `theta_1` (pendiente) y `coste_final`.

### 2. `gradiente_descendente(X, y, alpha=0.1, iteraciones=1000)`
Función de optimización core. Itera actualizando los parámetros en la dirección opuesta al gradiente de la función de coste.
*   **Parámetros:** Similares a `ajustar_modelo` (sin el parámetro gráfico).
*   **Retorna:** Una tupla `(theta_0, theta_1)` con los parámetros optimizados.

### 3. `funcion_coste(X, y, theta_0, theta_1)`
Calcula el Error Cuadrático Medio (MSE) para un conjunto de parámetros dado.
*   **Parámetros:** Datos `X`, `y` y los escalares `theta_0`, `theta_1`.
*   **Retorna:** Un flotante que representa el valor de coste $J(\theta_0, \theta_1)$.

### 4. `hipotesis(X, theta_0, theta_1)`
Calcula las predicciones teóricas basadas en un modelo lineal.
*   **Parámetros:** Datos `X` y los parámetros del modelo.
*   **Retorna:** Un array con los valores predichos $\hat{y}$.

### 5. `generar_graficos(X, y, theta_0, theta_1)`
Función auxiliar de visualización basada en `matplotlib`. Genera tres subgráficos: la superficie del paraboloide de coste, las curvas de nivel 2D con el mínimo marcado, y la dispersión de datos superpuesta con la recta obtenida.
