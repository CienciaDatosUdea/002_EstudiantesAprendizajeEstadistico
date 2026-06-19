# Documentacion de Uso

## API principal

### linear_hypothesis(x, theta0, theta1)
- Objetivo: calcular la hipotesis lineal h(x) = theta0 + theta1*x.
- Entradas:
  - x: arreglo o lista de valores de entrada.
  - theta0: intercepto del modelo.
  - theta1: pendiente del modelo.
- Salida: arreglo numpy con predicciones.

### compute_cost(x, y, theta0, theta1)
- Objetivo: calcular la funcion de coste cuadratica.
- Formula:
  J(theta0, theta1) = (1/(2m)) * sum((h(x)-y)^2)
- Entradas: x, y, theta0, theta1.
- Salida: costo escalar tipo float.

### gradient_descent(x, y, theta0_init, theta1_init, alpha, n_iter, tol, return_history)
- Objetivo: optimizar theta0 y theta1 minimizando la funcion de coste.
- Entradas clave:
  - alpha: tasa de aprendizaje.
  - n_iter: numero maximo de iteraciones.
  - tol: tolerancia para criterio de parada.
- Salida: diccionario con theta0, theta1, cost, iterations y history.

### fit_linear_regression(x, y, ...)
- Objetivo: funcion principal de alto nivel para entrenar el modelo.
- Salida: misma estructura de gradient_descent.

## Flujo recomendado

1. Cargar o definir X y y.
2. Llamar fit_linear_regression.
3. Revisar theta0, theta1 y costo final.
4. Usar linear_hypothesis para nuevas predicciones.
