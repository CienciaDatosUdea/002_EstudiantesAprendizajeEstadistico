# Síntesis del plan de análisis

Mateo Caballero Serna, Lucian Valle Aguilar

**Observación:** un evento (conglomerado × campaña mensual) con su total de *Anopheles* y su esfuerzo en persona-horas.

**Qué predecimos:** la tasa por persona-hora y, con ella, el total de 4 noches en Cheté (288 persona-horas) y la probabilidad de pasar 250, 500, 1 000 y 1 500 mosquitos.

**Cuándo es buena:** si le gana a la tasa histórica del conglomerado y a nuestro Fermi en error (MAE en escala log), y si su intervalo del 80 % contiene lo real cerca del 80 % de las veces.

**Datos:** todo el estudio para entrenar (879 eventos, 40 conglomerados, enero 2021 a marzo 2023) y Cheté aparte para evaluar. Comparamos con tasas por persona-hora porque las campañas duran distinto.

**Antes de confiar en un dato:** quitamos duplicados, corregimos años mal escritos, completamos fechas vacías, usamos 14 h donde el esfuerzo decía 7 h con horario de 14 h y unificamos los nombres de conglomerado. Especie ausente en un evento = 0; evento ausente = dato faltante.

**Lluvias:** no se puede comprobar con el archivo; haría falta lluvia del IDEAM o satelital. **Medición extra:** la lluvia de cada noche.

**No se puede responder:** adentro vs. afuera, hora de picadura, diferencias entre casas o noches, efecto de las intervenciones y fracción de *Anopheles*.

**Campañas reservadas:** desde el 1 de noviembre de 2022 (200 eventos, 23 %). Validación con origen móvil mensual, bootstrap por conglomerado o campaña y evaluación final una sola vez.
