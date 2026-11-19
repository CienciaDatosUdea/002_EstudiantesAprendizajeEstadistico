# Plan de análisis

Mateo Caballero Serna, Lucian Valle Aguilar

## Tres decisiones

**¿Qué es una observación?** Un evento: un conglomerado en una campaña mensual (por ejemplo `Chete1-05`), con su total de *Anopheles* y su esfuerzo en persona-horas (horas por noche × noches × 2 colectores). Es lo más fino que trae el archivo, porque cada fila suma todas las casas, noches y horas.

**¿Qué vamos a predecir?** La tasa de captura (*Anopheles* por persona-hora). Con ella damos el total esperado de una campaña de 4 noches en Cheté con nuestro diseño (288 persona-horas) y la probabilidad de pasar 250, 500, 1 000 y 1 500 mosquitos, que es lo que define cuántos frascos llevar.

**¿Cómo sabremos si funciona?** Si se equivoca menos que una referencia sencilla (la tasa histórica del conglomerado y nuestro Fermi) y si su intervalo del 80 % contiene lo real cerca del 80 % de las veces. Medimos el error en escala log (MAE de log10 de la tasa), porque los conteos van de unos pocos a miles.

## Preguntas

**¿Solo Cheté, todo Timbiquí o todo el estudio?** Todo el estudio para entrenar y Cheté aparte para evaluar. Con todo ganamos datos (Cheté solo tiene 34 eventos antes del corte), pero perdemos parecido: Cheté es casi todo *An. albimanus* y la zona casi todo *An. neivai*. Por eso el modelo usa la historia propia de cada conglomerado.

**¿Qué campañas se pueden comparar y cómo tratamos duraciones, lugares o especies distintas?** Usamos tasas por persona-hora y no totales, porque la línea base duró 14 horas por noche, el resto 12, y las noches van de 1 a 6. Las especies van juntas (total de *Anopheles*) salvo cuando la pregunta es sobre especies.

**¿Qué revisamos antes de confiar en un dato?** Duplicados, especies repetidas en un evento, fechas vacías o con el año mal escrito, esfuerzo que no coincide con el horario, nombres de conglomerado distintos, taxonomía, coordenadas y que los metadatos coincidan con el archivo.

**¿Cómo distinguimos "no se capturó" de "no se anotó"?** Si un evento existe y le falta una especie, esa especie cuenta como 0. Si falta el evento completo, es un dato faltante: no sabemos si no hubo campaña o si no se anotó.

**¿Vale la pena comprobar la temporada de lluvias?** Sí, pero el archivo no trae lluvia. Habría que cruzar las fechas con la lluvia diaria de una estación del IDEAM cercana o de datos satelitales.

**¿Qué medición extra tomaríamos?** La lluvia de cada noche, con un pluviómetro. Es lo que más podría explicar por qué la tasa cambia tanto de un mes a otro.

**¿Qué preguntas no se pueden responder?** Si se pica más adentro o afuera, a qué hora, cuánto cambia entre casas o noches, si funcionaron los mosquiteros o el rociado, y qué fracción de lo capturado es *Anopheles*. El archivo no trae esos campos.

**Una pregunta nuestra (punto 4):** ¿cuántas campañas hacen falta para conocer la tasa de un lugar con confianza?

## Campañas reservadas

Todos los eventos que empiezan desde el 1 de noviembre de 2022: 200 de 879 (23 %), 10 de ellos de Cheté. La regla se fijó mirando solo fechas y número de eventos por mes, y se aplica después de corregir los años mal escritos. La lista está en `campanas_reservadas.csv`.

## Validación

Origen móvil: para cada mes desde enero de 2022 entrenamos con lo anterior y predecimos ese mes. Las variables solo miran al pasado de cada conglomerado. El bootstrap remuestrea conglomerados o campañas, no filas, porque las visitas al mismo lugar no son independientes. Las campañas reservadas se usan una sola vez, al final.

## Qué cambió y por qué

- Punto 1: pasamos de una tabla de "relevancia" con pesos que sumaban 100 a factores que se multiplican, porque los pesos no daban un número de mosquitos.
- Punto 1: dejamos de estimar la población del pueblo y partimos de una tasa por persona-hora, porque la población no se puede medir.
- Punto 1: bajamos de 40 casas con 3 colectores a 3 casas con 1 adentro y 1 afuera, porque 120 colectores no es realista.
- Plan: la observación pasó de casa-noche a evento, porque el archivo viene sumado por evento.
- Plan: corregimos los años mal escritos antes de fijar el corte; si no, campañas de 2023 habrían quedado en el entrenamiento.
- Punto 2: el hallazgo pasó de "Cheté es único" a "Cheté y Coteje forman un grupo", porque la permutación mostró que Coteje es igual.
- Punto 3: en validación ganó la binomial negativa, pero en las campañas reservadas ganó la tasa histórica, así que la predicción final usa la histórica.

## Consultas a la IA que cambiaron decisiones

- Cambiar los pesos por factores multiplicativos y usar una tasa por persona-hora (punto 1).
- Revisar la estructura del archivo: cada fila es un evento sumado y hay años mal escritos (punto 2).
- Hacer el bootstrap y la validación por conglomerado o campaña, no por fila (puntos 2 y 3).
- Al revisar duplicados con la IA, antes de separar las campañas, se vieron los conteos de un evento reservado (`Chete2-20`). No se usó para ninguna decisión.
