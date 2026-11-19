# Reto Final: Plan de Análisis

**Curso:** Aprendizaje Estadístico 2026-2  
**Estudiantes:** Jeronimo Lopez Gomez - Jose Daniel Godoy

---

## 1. Pregunta inicial

**¿Cuántos individuos del género *Anopheles* se capturarían en una próxima campaña de cuatro noches en Cheté y con qué incertidumbre?**

Partimos de nuestra estimación de Fermi, realizada antes de conocer los datos históricos, en la cual estimamos un total de **532 mosquitos**, de los cuales aproximadamente **346 serían del género *Anopheles***, con escenarios mínimos y máximos de 122 y 605 individuos, respectivamente.

Es importante aclarar que este rango corresponde a escenarios hipotéticos construidos a partir de nuestros supuestos iniciales y no representa un intervalo de confianza estadístico.

Nuestro propósito será utilizar los registros históricos de captura para evaluar qué tan razonables fueron dichos supuestos y obtener una nueva estimación sustentada en los datos.

## 2. Unidad de observación y población de estudio

Inicialmente, consideramos como observación cada individuo capturado, identificado mediante su evento de recolección, fecha, hora, ubicación y colector.

Para comparar las capturas, agruparemos los registros por fecha, lugar y condiciones de muestreo, teniendo en cuenta el número de individuos capturados y el tiempo efectivo de recolección cuando esta información esté disponible.

Consideramos una campaña de cuatro noches con el siguiente protocolo preliminar, basado en nuestra estimación de Fermi:

- **Ubicación:** Corregimiento de Cheté, municipio de Timbiquí, Cauca (2°39'32.35"N, 77°37'24.61"W).
- **Puntos de captura:** Cinco puntos de recolección, distribuidos entre dos interiores y tres exteriores de las viviendas.
- **Duración:** Cinco horas efectivas por noche, aproximadamente entre las 6:00 p. m. y las 11:00 p. m.
- **Franjas horarias:** Dos intervalos, de 6:00 p. m. a 9:00 p. m. y de 9:00 p. m. a 11:00 p. m., para estudiar posibles diferencias en las capturas.
- **Registro de capturas:** Fecha, hora, coordenadas, ambiente, punto de recolección, colector e identificación del individuo.
- **Condiciones ambientales:** Temperatura, humedad relativa y precipitación, cuando sea posible medirlas. Como información complementaria utilizaremos registros de estaciones meteorológicas del IDEAM cercanas a Cheté.
- **Esfuerzo de muestreo:** Se registrarán las horas efectivas de captura, incluyendo posibles interrupciones.
- **Identificación:** Los individuos serán identificados según los procedimientos establecidos por el equipo de entomología.

El interés principal será estudiar los mosquitos del género *Anopheles*, considerando las diferentes especies registradas.

Este protocolo es una propuesta inicial y podrá compararse con las condiciones reales de las campañas históricas.

## 3. Fuentes de información

**Base de datos Darwin Core (`occurrence.txt`):**

Utilizaremos los registros de capturas del género *Anopheles* proporcionados por el INS. Revisaremos principalmente la información temporal y geográfica de las capturas, la identificación de especies, los conteos y el esfuerzo de recolección.

También consultaremos la descripción del conjunto de datos para comprender cómo se realizaron las capturas y qué tan comparables son con las condiciones que supusimos inicialmente.

**Bases de datos DHIME-IDEAM (`dhime_mensual.csv` y `dhime_diario.csv`):**

Utilizaremos los registros de precipitación de estaciones meteorológicas de la región. La estación más cercana identificada se encuentra aproximadamente a 12.9 km al noroeste de Cheté.

Analizaremos la precipitación en milímetros y su distribución temporal, teniendo en cuenta la ubicación de las estaciones y la disponibilidad de los registros.

**Hoja de campo:**

Diseñaremos una hoja para registrar cada captura y sus condiciones de recolección. Posteriormente, compararemos sus variables con las disponibles en el conjunto de datos histórico para identificar información adicional o faltante. La hoja de campo se establece en el documento `hoja_de_campo.md`

**Notebook de análisis exploratorio (`EDA.ipynb`):**

Realizaremos un análisis exploratorio de los registros del INS para comprender su estructura, identificar valores faltantes y duplicados, revisar la distribución geográfica y temporal de las capturas, el esfuerzo de muestreo y las especies registradas.

También estudiaremos la precipitación histórica y construiremos agrupaciones de cuatro jornadas de captura como una primera aproximación al escenario planteado en nuestra estimación de Fermi.

## 4. Revisión de la calidad de los datos

Antes de realizar comparaciones y estimaciones, revisaremos:

- Registros duplicados, valores faltantes y posibles errores en fechas o coordenadas.
- Disponibilidad de información sobre el número de individuos y el tiempo de recolección.
- Diferencias entre lugares, horarios y periodos de muestreo.
- Posibles cambios en el procedimiento de captura durante los años del estudio.
- Disponibilidad de registros de precipitación para las fechas de recolección.

Además, diferenciaremos entre una jornada en la que no se capturaron mosquitos y una jornada de la cual no se dispone de información, cuando los registros permitan hacerlo.

## 5. Preguntas exploratorias

A partir de las variables disponibles y de los análisis planteados en el notebook, buscaremos responder las siguientes preguntas:

1. **¿Cómo se distribuyen geográfica y temporalmente los registros de captura del género *Anopheles*?** Identificaremos municipios, localidades, fechas y periodos con mayor cantidad de registros, prestando especial atención a Cheté.

2. **¿Cuáles son las especies de *Anopheles* más registradas en la zona de estudio?** Analizaremos la frecuencia de las distintas especies y su distribución en las localidades.

3. **¿Cómo varía el número de mosquitos registrados entre los diferentes meses y jornadas de recolección?** Estudiaremos la variación temporal y las posibles diferencias entre periodos.

4. **¿Qué relación se observa entre la precipitación y el número de mosquitos capturados?** Compararemos los registros del IDEAM con las fechas y cantidades de mosquitos recolectados para identificar posibles patrones.

5. **¿Cuántos mosquitos se registran al agrupar cuatro jornadas de recolección en Cheté?** Compararemos estas agrupaciones con nuestra estimación inicial de Fermi, teniendo en cuenta las diferencias de duración y esfuerzo entre las jornadas.

Estas preguntas permitirán identificar patrones en los datos históricos y establecer cuáles de nuestros supuestos iniciales pueden contrastarse con la información disponible.

## 6. Comparación con la estimación de Fermi

Retomaremos nuestra estimación inicial de **346 individuos de *Anopheles* en cuatro noches** y la compararemos con las capturas históricas disponibles para Cheté.

Buscaremos determinar:

- Si las cantidades registradas se aproximan a nuestro escenario central o a los escenarios mínimo y máximo.
- Si las horas y los lugares de recolección considerados inicialmente son comparables con los del estudio.
- Si existen diferencias importantes entre capturas interiores y exteriores.
- Si la precipitación parece estar relacionada con las variaciones en las capturas.

Calcularemos las diferencias entre la estimación inicial y los valores obtenidos a partir de los registros históricos, procurando comparar jornadas con condiciones de recolección similares.

## 7. Estimación de incertidumbre y predicción

Nuestro objetivo será estimar el número de individuos del género *Anopheles* que podrían capturarse durante una próxima campaña de cuatro noches en Cheté, considerando principalmente la información histórica de capturas, el esfuerzo de recolección y las condiciones de precipitación.

Para ello, proponemos utilizar las siguientes herramientas del curso:

- **Estadística descriptiva:** Para estudiar promedios, estadisticas de dispersión y diferencias entre jornadas de captura.
- **Bootstrap:** Para estimar la incertidumbre de las capturas promedio, tomando como referencia jornadas o campañas comparables y evitando tratar observaciones repetidas como independientes.
- **Regresión:** Se propone construir un modelo sencillo que permita estimar el número de mosquitos capturados a partir de variables disponibles, como el periodo del año, el esfuerzo de recolección y la información de precipitación conocida antes de la campaña.

Como referencia, compararemos los resultados del modelo con nuestra estimación de Fermi revisada y con una estimación sencilla basada en el promedio histórico de capturas.

### Evaluación del modelo

Para evaluar el desempeño del modelo, utilizaremos medidas de error como RMSE o MAE que permitan medir qué tan alejadas están las predicciones de las cantidades observadas.

La evaluación respetará el orden temporal de los registros: utilizaremos periodos anteriores para construir el modelo y reservaremos desde el inicio las campañas más recientes para la evaluación final, sin utilizarlas para ajustar los resultados.

También tendremos en cuenta que las capturas repetidas en un mismo lugar o campaña pueden estar relacionadas entre sí.

Consideraremos que el modelo resulta útil si permite estimar las capturas con un error menor al de una referencia histórica sencilla y si su incertidumbre ayuda a planificar la cantidad de materiales necesarios para una próxima campaña.

La construcción y evaluación del modelo dependerán de la cantidad y calidad de los registros disponibles. Si los datos no permiten una predicción confiable, explicaremos sus limitaciones y presentaremos los resultados exploratorios y las estimaciones que sí puedan justificarse.

## 8. Pregunta propia

**¿Las capturas de mosquitos del género *Anopheles* presentan diferencias entre el interior y el exterior de las viviendas?**

Nos interesa conocer si la ubicación de los puntos de recolección influye en la cantidad de individuos registrados.

Para responder esta pregunta, compararemos las capturas clasificadas por ambiente y, cuando sea posible, tendremos en cuenta las horas de recolección para evitar comparar jornadas con esfuerzos muy diferentes.

Este análisis permitirá evaluar uno de los supuestos que planteamos en nuestra estimación de Fermi, pero es claro que con los datos que contamos al momento no es posible estimar dicho supuesto, sin embargo, se propone poder modificar las siguientes campañas para que se considere este factor.

## 9. Limitaciones y cambios del plan

Reconocemos que existen algunas limitaciones en los datos y en las comparaciones que pretendemos realizar.

En particular, los registros históricos pueden presentar diferencias en los tiempos de recolección, lugares de muestreo y disponibilidad de información. Además, los registros de precipitación de estaciones meteorológicas cercanas no necesariamente representan las condiciones exactas en cada punto de captura.

Las agrupaciones de cuatro jornadas que construimos en el notebook representan una agrupación inicial y no necesariamente corresponden a campañas de cuatro noches consecutivas.

Los datos sobre la ubicación de los colectores no se tiene en cuenta en los datos de apoyo por lo cuál será complejo poder estimar con estas herramientas, la influencia de la ubicación de los colectores en la tasa de captura efectiva.

El plan podrá modificarse a medida que avancemos en el análisis. Registraremos los cambios realizados, sus motivos y su efecto sobre las decisiones iniciales, manteniendo intacta nuestra estimación original de Fermi.

---

**Registro de modificaciones del plan**

| Fecha | Modificación | Justificación |
|---|---|---|
| Por completar | Elaboración del plan inicial | Definir las preguntas y métodos de análisis antes de completar el estudio |