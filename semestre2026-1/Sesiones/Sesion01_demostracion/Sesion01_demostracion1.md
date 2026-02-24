Actúa como un ingeniero senior de simulación científica en Python. Quiero un script COMPLETO (un solo archivo) que simule un PÉNDULO DOBLE y lo ANIME.

Requisitos funcionales:
1) Modelo físico:
   - Usa el modelo clásico de doble péndulo plano con dos masas puntuales m1, m2 y varillas rígidas sin masa de longitudes L1, L2.
   - Variables de estado: θ1, θ2 (ángulos desde la vertical hacia abajo) y sus velocidades angulares ω1, ω2.
   - Gravedad g constante.
   - Incluye una opción de amortiguamiento viscoso lineal (b1, b2) sobre ω1 y ω2 (si b=0, sin amortiguamiento).
   - Deriva o implementa las ecuaciones estándar (acopladas y no lineales) y explícitalas en comentarios.

2) Integración numérica:
   - Implementa integración con SciPy `solve_ivp` usando método `DOP853` o `RK45`.
   - Permite definir `t_max` y `dt` (para muestrear la solución en `t_eval`).
   - Agrega chequeos básicos de estabilidad: si la solución explota (NaN/Inf), detén y muestra un mensaje.
   - Calcula y grafica la energía total (T+V) para validar (en el caso b1=b2=0 la energía debe ser casi constante; muestra el error relativo).

3) Animación:
   - Usa `matplotlib.animation.FuncAnimation`.
   - Debe mostrar:
     a) el péndulo (dos segmentos),
     b) las masas (puntos),
     c) un rastro (trail) del segundo bob (configurable: longitud del rastro en segundos o número de puntos).
   - Ajusta límites automáticamente para que siempre se vea completo (p. ej. ±(L1+L2)*1.1).
   - Debe poder correr en un notebook y también como script normal.
   - Opción para guardar como MP4 o GIF si el usuario lo desea (con instrucciones claras y try/except si falta ffmpeg).

4) Interactividad / parámetros:
   - Define parámetros al inicio del script: m1, m2, L1, L2, g, b1, b2, condiciones iniciales (θ1, θ2, ω1, ω2), t_max, dt, trail_length.
   - Incluye un bloque `if __name__ == "__main__":` que ejecute todo.
   - Incluye una función `simulate(params)->results` y una función `animate(results, params)` para mantener modularidad.

5) Calidad de código:
   - Código limpio, con docstrings, type hints, y comentarios mínimos pero útiles.
   - Evita magia: nombra bien las variables y separa “modelo” de “visualización”.
   - No uses seaborn. Solo numpy, scipy, matplotlib.
   - Incluye una explicación breve (10-15 líneas) al inicio del archivo sobre el modelo, las unidades y cómo ajustar parámetros.

Extra (si puedes):
- Agrega una opción para comparar dos trayectorias con condiciones iniciales casi iguales (sensibilidad al caos): anima ambas con colores distintos y muestra distancia angular |θ1a-θ1b|+|θ2a-θ2b| en un subplot pequeño.

Entrega:
- Devuélveme SOLO el código Python final en un bloque ```python``` listo para copiar y ejecutar.

