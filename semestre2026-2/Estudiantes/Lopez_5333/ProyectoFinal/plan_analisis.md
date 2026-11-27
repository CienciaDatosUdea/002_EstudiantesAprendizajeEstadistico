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

Inicialmente consideramos como observación cada individuo capturado. Al revisar los datos, vimos que cada registro de `occurrence.txt` corresponde a una especie en una casa durante una campaña mensual y acumula varias noches. Por eso definimos:

- **Visita:** una casa en una campaña mensual. Se suman las especies registradas.
- **Esfuerzo:** noches × horas por noche (tomado de `samplingEffort`), en **casa-horas**. Suponemos un colector por casa, porque el dataset no informa cuántos había.
- **Tasa de captura:** *Anopheles* por casa-hora = individuos / esfuerzo. Para varias casas o un mes se usa la razón de sumas (Σ individuos / Σ casa-horas).
- **Campaña:** las casas de una localidad en un mismo mes. Para el modelo, la observación es la **campaña de Cheté** (sus dos casas sumadas), porque las casas de un mismo mes no son independientes.

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

Este protocolo es una propuesta inicial y podrá compararse con las condiciones reales de las campañas históricas. En los datos históricos, todas las capturas se hicieron dentro de la vivienda ("En domicilio"), con cebo humano y durante la noche completa (17:00 a 05:00, unas 12 h). No hay registros exteriores ni por franja horaria.

## 3. Fuentes de información

**Base de datos Darwin Core (`occurrence.txt`):**

Utilizaremos los registros de capturas del género *Anopheles* proporcionados por el INS. Revisaremos principalmente la información temporal y geográfica de las capturas, la identificación de especies, los conteos y el esfuerzo de recolección. Trabajaremos solo con los municipios de **Guapi y Timbiquí** (Cauca) y descartaremos los registros de Santa Bárbara (Nariño), que quedan fuera de la zona de estudio. El mes de cada campaña se tomará de `verbatimEventDate`, porque `eventDate` tiene valores faltantes y errores de año.

También consultaremos la descripción del conjunto de datos para comprender cómo se realizaron las capturas y qué tan comparables son con las condiciones que supusimos inicialmente.

**Bases de datos DHIME-IDEAM (`dhime_mensual.csv` y `dhime_diario.csv`):**

Utilizaremos los registros de precipitación de estaciones meteorológicas de la región. La estación más cercana identificada se encuentra aproximadamente a 12.9 km al noroeste de Cheté.

Analizaremos la precipitación en milímetros y su distribución temporal, teniendo en cuenta la ubicación de las estaciones y la disponibilidad de los registros.

**Hoja de campo:**

Diseñaremos una hoja para registrar cada captura y sus condiciones de recolección. Posteriormente, compararemos sus variables con las disponibles en el conjunto de datos histórico para identificar información adicional o faltante. La hoja de campo se establece en el documento `hoja_de_campo.md`

**Notebook de análisis exploratorio (`EDA.ipynb`):**

Realizaremos un análisis exploratorio de los registros del INS para comprender su estructura, identificar valores faltantes y duplicados, revisar la distribución geográfica y temporal de las capturas, el esfuerzo de muestreo y las especies registradas.

Normalizaremos las capturas por esfuerzo (*Anopheles* por casa-hora) y estudiaremos la precipitación por campaña. También compararemos Cheté con las localidades vecinas de la ribera del río Timbiquí (Coteje, Velásquez y San Miguel del Río) y validaremos la hipótesis **H1: la captura es proporcional al esfuerzo**.

Inicialmente planeábamos agrupar fechas de recolección de a cuatro. Lo descartamos porque esas agrupaciones mezclaban localidades y periodos de varios meses; la campaña mensual ya representa las varias noches de muestreo.

**Notebook del modelo (`model.ipynb`):**

Construiremos el modelo predictivo para Cheté a partir de H1, lo evaluaremos respetando el orden temporal y estimaremos la próxima campaña con su incertidumbre.

## 4. Revisión de la calidad de los datos

Antes de realizar comparaciones y estimaciones, revisaremos:

- Registros duplicados, valores faltantes y posibles errores en fechas o coordenadas.
- Disponibilidad de información sobre el número de individuos y el tiempo de recolección.
- Diferencias entre lugares, horarios y periodos de muestreo.
- Posibles cambios en el procedimiento de captura durante los años del estudio.
- Disponibilidad de registros de precipitación para las fechas de recolección.

Además, diferenciaremos entre una jornada en la que no se capturaron mosquitos y una jornada de la cual no se dispone de información, cuando los registros permitan hacerlo. En `occurrence.txt` solo hay registros cuando hubo capturas, así que las visitas con cero individuos no se pueden identificar.

## 5. Preguntas exploratorias

A partir de las variables disponibles y de los análisis planteados en el notebook, buscaremos responder las siguientes preguntas:

1. **¿Cómo se distribuyen geográfica y temporalmente los registros de captura del género *Anopheles*?** Identificaremos municipios, localidades, fechas y periodos con mayor cantidad de registros, prestando especial atención a Cheté.

2. **¿Cuáles son las especies de *Anopheles* más registradas en la zona de estudio?** Analizaremos la frecuencia de las distintas especies y su distribución en las localidades.

3. **¿Cómo varía el número de mosquitos registrados entre los diferentes meses y campañas?** Estudiaremos la variación temporal de la tasa de captura (*Anopheles* por casa-hora), para que las diferencias de esfuerzo entre campañas no se confundan con diferencias en la cantidad de mosquitos.

4. **¿Qué relación se observa entre la precipitación y el número de mosquitos capturados?** Correlacionaremos (Spearman) la lluvia de cada mes de muestreo con la cantidad de mosquitos, el esfuerzo y la tasa de captura. Consideraremos tres ventanas de lluvia: durante la campaña, 14 días antes y 30 días antes.

5. **¿La captura es proporcional al esfuerzo (H1)?** Estimaremos la pendiente entre capturas y esfuerzo en escala log-log. Una pendiente cercana a 1 respalda la estructura tasa × esfuerzo de nuestra estimación de Fermi.

6. **¿Las localidades vecinas de la ribera del río Timbiquí son equivalentes a Cheté?** Compararemos Cheté con Coteje, Velásquez y San Miguel del Río en tasa de captura (pareando por campaña) y en composición de especies, para decidir si sus datos pueden sumarse a los de Cheté.

Estas preguntas permitirán identificar patrones en los datos históricos y establecer cuáles de nuestros supuestos iniciales pueden contrastarse con la información disponible.

## 6. Comparación con la estimación de Fermi

Retomaremos nuestra estimación inicial de **346 individuos de *Anopheles* en cuatro noches** y la compararemos con las capturas históricas disponibles para Cheté.

Buscaremos determinar:

- Si las cantidades registradas se aproximan a nuestro escenario central o a los escenarios mínimo y máximo.
- Si las horas y los lugares de recolección considerados inicialmente son comparables con los del estudio.
- Si existen diferencias importantes entre capturas interiores y exteriores (no se puede verificar con los datos históricos, que solo tienen capturas dentro de la vivienda).
- Si la precipitación parece estar relacionada con las variaciones en las capturas.

Como las campañas históricas y nuestro diseño tienen esfuerzos distintos (2 casas × 12 h frente a 5 colectores × 5 h), la comparación principal se hará por **tasa horaria**: la tasa implícita en Fermi (346 / 100 colector-horas ≈ 3,46 *Anopheles* por colector-hora) frente a la tasa observada en Cheté. Después llevaremos esa tasa al esfuerzo de nuestro diseño.

## 7. Estimación de incertidumbre y predicción

Nuestro objetivo será estimar el número de individuos del género *Anopheles* que podrían capturarse durante una próxima campaña de cuatro noches en Cheté, considerando principalmente la información histórica de capturas y el esfuerzo de recolección; la precipitación se estudia de forma exploratoria en el EDA.

Para ello, proponemos utilizar las siguientes herramientas del curso:

- **Estadística descriptiva:** Para estudiar promedios, estadisticas de dispersión y diferencias entre jornadas de captura.
- **Bootstrap:** Para estimar la incertidumbre, remuestreando **campañas completas** (no visitas sueltas), para no tratar observaciones repetidas como independientes.
- **Regresión:** Una regresión lineal **sin intercepto** basada en H1, $\hat N = \beta \times \text{colector-horas}$, ajustada sobre las campañas de Cheté. La pendiente $\beta$ es la tasa de captura (*Anopheles* por colector-hora): es la estimación de Fermi con la tasa calibrada con datos (**Fermi revisada**). Se usa solo el esfuerzo, que el equipo conoce antes de salir. La precipitación no se incluye porque en el EDA su relación con la tasa fue débil y no significativa.

Como referencia, compararemos los resultados del modelo con nuestra estimación de Fermi original (tasa de 3,46 × esfuerzo).

Para la próxima campaña (diseño de Fermi: 5 colectores × 5 h × 4 noches = 100 colector-horas) reportaremos:
- la predicción $\hat\beta \times 100$;
- el IC 95 % del valor esperado (bootstrap de campañas);
- el rango del 80 % de una campaña concreta, obtenido aplicando la tasa de cada campaña histórica al diseño.

### Evaluación del modelo

Para evaluar el desempeño del modelo, utilizaremos medidas de error como RMSE o MAE que permitan medir qué tan alejadas están las predicciones de las cantidades observadas.

La evaluación respetará el orden temporal de los registros: utilizaremos periodos anteriores para construir el modelo y reservaremos desde el inicio las campañas más recientes para la evaluación final, sin utilizarlas para ajustar los resultados.

- **Reservadas:** las 4 campañas más recientes de Cheté (diciembre de 2022 a marzo de 2023).
- **Validación:** sobre las 18 campañas restantes, cada campaña a partir de la 11.ª se predice con un modelo ajustado solo con las campañas anteriores.
- **Medida de error:** MAE, comparado contra Fermi original.

También tendremos en cuenta que las capturas repetidas en un mismo lugar o campaña pueden estar relacionadas entre sí.

Consideraremos que el modelo resulta útil si permite estimar las capturas con un error menor al de la estimación de Fermi original y si su incertidumbre ayuda a planificar la cantidad de materiales necesarios para una próxima campaña.

La construcción y evaluación del modelo dependerán de la cantidad y calidad de los registros disponibles. Si los datos no permiten una predicción confiable, explicaremos sus limitaciones y presentaremos los resultados exploratorios y las estimaciones que sí puedan justificarse.

## 8. Pregunta propia

**¿Las capturas de mosquitos del género *Anopheles* presentan diferencias entre el interior y el exterior de las viviendas?**

Nos interesa conocer si la ubicación de los puntos de recolección influye en la cantidad de individuos registrados.

Para responder esta pregunta, compararemos las capturas clasificadas por ambiente y, cuando sea posible, tendremos en cuenta las horas de recolección para evitar comparar jornadas con esfuerzos muy diferentes.

Este análisis permitirá evaluar uno de los supuestos que planteamos en nuestra estimación de Fermi, pero es claro que con los datos que contamos al momento no es posible estimar dicho supuesto, sin embargo, se propone poder modificar las siguientes campañas para que se considere este factor.

## 9. Limitaciones y cambios del plan

Reconocemos que existen algunas limitaciones en los datos y en las comparaciones que pretendemos realizar.

En particular, los registros históricos pueden presentar diferencias en los tiempos de recolección, lugares de muestreo y disponibilidad de información. Además, los registros de precipitación de estaciones meteorológicas cercanas no necesariamente representan las condiciones exactas en cada punto de captura.

Las noches de una campaña no son necesariamente consecutivas: pueden repartirse en dos o tres semanas del mismo mes.

Suponemos un colector por casa; si hubiera más, la tasa por colector-hora sería menor. Además, la tasa histórica se midió con unas 12 h por noche y se aplicará a las 5 h de nuestro diseño, aunque la actividad de los mosquitos puede no ser igual a lo largo de la noche.

Cheté tiene solo 2 casas y 22 campañas, y las casas difieren mucho entre sí: la elección de las casas es una fuente de incertidumbre importante.

Los datos sobre la ubicación de los colectores no se tiene en cuenta en los datos de apoyo por lo cuál será complejo poder estimar con estas herramientas, la influencia de la ubicación de los colectores en la tasa de captura efectiva.

El plan podrá modificarse a medida que avancemos en el análisis. Registraremos los cambios realizados, sus motivos y su efecto sobre las decisiones iniciales, manteniendo intacta nuestra estimación original de Fermi.

---

**Registro de modificaciones del plan**

| Fecha | Modificación | Justificación |
|---|---|---|
| Por completar | Elaboración del plan inicial | Definir las preguntas y métodos de análisis antes de completar el estudio |
| 2026-11-23 | La observación pasa a ser la visita (casa × mes) y la campaña; la captura se mide como *Anopheles* por casa-hora | Cada registro acumula varias noches y el esfuerzo varía entre visitas (1 a 6 noches, 12 o 14 h) |
| 2026-11-23 | Se descartan las agrupaciones de cuatro fechas de recolección | Mezclaban localidades y periodos de varios meses; la campaña mensual ya representa las noches de muestreo |
| 2026-11-23 | Se descartan los registros de Santa Bárbara (Nariño) | El estudio se centra en Guapi y Timbiquí |
| 2026-11-23 | Se agregan H1 (captura proporcional al esfuerzo) y la comparación con las localidades vecinas de la ribera del río Timbiquí | Son los supuestos de Fermi que sí se pueden verificar con los datos |
| 2026-11-23 | El modelo es una regresión lineal sin intercepto sobre el esfuerzo, sin precipitación ni periodo del año | H1 se sostiene en el EDA y la relación de la lluvia con la tasa fue débil y no significativa |
| 2026-11-23 | La referencia del modelo es Fermi original en lugar del promedio histórico | El objetivo del reto es revisar nuestra estimación de Fermi con los datos |