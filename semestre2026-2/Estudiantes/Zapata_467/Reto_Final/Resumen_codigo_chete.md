# Análisis de capturas de mosquitos en Cheté
## ¿Qué hicimos en el código y qué falta para completar la rúbrica?

**Proyecto:** *Cuatro noches para contar mosquitos*  
**Documento de trabajo:** síntesis de avances y pendientes  
**Base de revisión:** `Analisis_Chete_01 (2).ipynb`, `Estimación de Fermi_00_Zapata_Barrera(1).md`, `Bitacora_Analisis_Chete_01(1).md` y `RetoV2 (1).pdf`.  
**Criterio:** distinguir lo ejecutado en el notebook, lo documentado en los archivos y lo que todavía debe comprobarse. Las propuestas de la IA no se atribuyen automáticamente al equipo.

---

# 1. ¿Qué hicimos en el código?

## 1.1. Preparación y limpieza de los datos

1. Importamos el conjunto original de datos utilizando **pandas**.
2. Seleccionamos únicamente los registros correspondientes a **Cheté**: 77 registros de un dataset de 1.502 filas.
3. Identificamos las columnas principales del análisis: `eventID`, `eventDate`, `individualCount` y `samplingEffort`.
4. Revisamos valores faltantes, la cantidad de eventos y las repeticiones de identificadores.
5. Completamos la fecha faltante del registro `Chete1-20` (349 mosquitos) a partir de otro registro del mismo `eventID`, conservando una marca de imputación. Esta asignación es una **decisión metodológica**, no una fecha observada directamente.
6. Extraímos el número de noches del texto de `samplingEffort` y detectamos diferencias entre filas de algunos eventos.
7. Agrupamos por `eventID`, sumamos los mosquitos de las especies registradas y contabilizamos el esfuerzo una sola vez por evento. Ante diferencias en las noches de un mismo evento, usamos el **máximo reportado**.
8. Obtuvimos una tabla intermedia con **44 eventos** y calculamos las capturas por **sitio-noche**.

## 1.2. Análisis exploratorio

9. Asignamos cada evento a un mes a partir de la fecha inicial registrada y agrupamos los eventos por mes.
10. Calculamos las capturas mensuales, las sitio-noches correspondientes y la tasa de **mosquitos por sitio-noche**. Esta tasa es `total de mosquitos / total de sitio-noches`, no el promedio simple de tasas de eventos.
11. Graficamos las tasas mensuales para describir su evolución a través de los años disponibles.
12. Incluimos explícitamente los meses sin muestreo como valores **faltantes**, no como ceros.
13. Calculamos los cambios absolutos y porcentuales de la tasa entre **meses calendario consecutivos con información**.
14. Graficamos la variación porcentual mensual para identificar aumentos y disminuciones.
15. Comparamos el orden de magnitud de las capturas con nuestra **estimación de Fermi**, observando que los protocolos y esfuerzos considerados no eran directamente equivalentes.
16. Extrajimos las **horas por noche** y calculamos sitio-horas para comparar campañas de distinta duración.
17. Calculamos las tasas de captura mensuales por sitio-hora y una tasa histórica global de aproximadamente **4,16 mosquitos/sitio-hora**.
18. Investigamos la correlación lineal de Pearson entre la tasa mensual actual y la de **uno y dos meses calendario anteriores**; obtuvimos, respectivamente, `r = -0,254` con 17 pares y `r = 0,190` con 16 pares. Son asociaciones exploratorias; no demuestran causalidad ni capacidad predictiva.

## 1.3. Construcción y evaluación de modelos predictivos

19. La **IA propuso** un primer modelo sencillo (**modelo A**) basado en la tasa histórica de capturas por sitio-hora; el equipo lo implementó y examinó.
20. Definimos como escenario operativo una campaña de **4 noches × 12 horas/noche × 2 sitios = 96 sitio-horas**.
21. El modelo A predijo aproximadamente **400 mosquitos** para ese escenario.
22. La **IA propuso** un segundo modelo (**modelo B**) de **regresión binomial negativa** con exposición en horas y una variable indicadora del sitio de captura; el equipo lo implementó.
23. Evaluamos ambos modelos mediante una validación temporal retrospectiva: en cada iteración se entrenó con los meses observados anteriores y se predijo el siguiente mes observado, manteniendo juntos los dos sitios de cada mes.
24. Calculamos el **error absoluto medio (MAE)** y el **error porcentual absoluto ponderado (WAPE)** en 13 meses de evaluación.
25. Comparamos los resultados: modelo A, **MAE = 122,72** y **WAPE = 32,73 %**; modelo B, **MAE = 119,04** y **WAPE = 31,74 %**.
26. Graficamos las capturas observadas y las predicciones de ambos modelos. La mejora del modelo B es **pequeña**; no demuestra una ventaja sustancial.

## 1.4. Cuantificación de la incertidumbre

27. Realizamos **30.000 remuestreos bootstrap de bloques mensuales** conservando juntos el conteo y el esfuerzo de cada mes.
28. Estimamos intervalos de confianza del 95 % para la tasa histórica: aproximadamente **[3,41; 4,93] mosquitos/sitio-hora** y **[41,31; 59,79] mosquitos/sitio-noche**.
29. Propagamos mediante bootstrap la incertidumbre de la tasa a la **media esperada de capturas** de la campaña futura, obteniendo un IC del 95 % aproximado de **[327; 473] mosquitos**.
30. Incorporamos una simulación de conteos con **sobredispersión**, mediante una binomial negativa cuando corresponde y Poisson cuando la sobredispersión estimada es nula.
31. Calculamos para el **modelo A** un intervalo predictivo del 95 % aproximado de **[132; 782] mosquitos** para la futura campaña de 96 sitio-horas.

## 1.5. Interpretación y pregunta propia

32. Identificamos que los resultados de los modelos no pueden compararse sin más con los **160 mosquitos** de Fermi: los escenarios difieren en esfuerzo y procedimiento de captura.
33. Encontramos diferencias entre las capturas asociadas a Chete1 y Chete2; el modelo B incorpora el sitio, pero no explica la causa ambiental de esa diferencia.
34. Retomamos nuestra hipótesis sobre **depósitos de agua posiblemente vinculados con excavaciones mineras**. Los datos actuales no incluyen una variable de actividad minera por fecha ni una medición verificada de esos depósitos, por lo que **no permiten calcular una correlación minería–capturas**.

---

# 2. Resultados clave que deben poder explicar en la sustentación

| Resultado | Interpretación correcta |
|---|---|
| 4,16 mosquitos/sitio-hora | Tasa histórica estimada a partir de capturas y esfuerzo; **no** equivale al tamaño de la población de mosquitos. |
| 400 mosquitos | Predicción puntual del modelo A para **96 sitio-horas**, no para cualquier diseño de cuatro noches. |
| IC 95 % [327; 473] | Incertidumbre estimada en la **media de capturas esperada** bajo el modelo A. |
| IP 95 % [132; 782] | Rango predictivo para una **campaña individual**, bajo los supuestos del procedimiento de simulación. |
| WAPE A = 32,73 %; B = 31,74 % | El modelo B mejora modestamente el error predictivo retrospectivo. |
| Pearson de uno y dos meses anteriores | Los coeficientes observados son pequeños y no sustentan una relación temporal lineal fuerte. |

**Limitaciones que deben declararse:** el bootstrap mensual independiente no captura necesariamente la dependencia temporal; dos visitas al mismo sitio no son réplicas independientes; la regresión necesita diagnóstico de convergencia y las horas de muestreo futuras deben interpretarse como **horas previstas**. Además, las campañas ya exploradas no constituyen un conjunto final verdaderamente intacto.

---

# 3. ¿Qué falta para completar la rúbrica del PDF?

La evaluación tiene **60 puntos grupales y 40 individuales**. El estado «por verificar» significa que no se encontró evidencia suficiente **en los archivos adjuntos**, no que el equipo no lo haya hecho.

| Criterio de la rúbrica | Puntos | Estado | Acción concreta pendiente |
|---|---:|---|---|
| **Equipo: antes de empacar** | 8 | **Parcial** | La estimación de Fermi incluye una cifra (**160**) y supuestos; añadir o localizar el **rango**, identificar el **supuesto que más pesa** y comprobar si la entrega se registró antes de consultar los datos. No presentar retrospectivamente una estimación como previa. |
| **Equipo: diseño y calidad de los datos** | 12 | **Parcial** | Aportar la **hoja de campo diseñada por el equipo**, compararla explícitamente con lo que registra el estudio, explicar origen/calidad de los datos y justificar las decisiones sobre fecha imputada, esfuerzo inconsistente y horas. |
| **Equipo: lo que cuentan los datos** | 12 | **Avanzado; pendiente cierre** | Elegir **un hallazgo principal** y someterlo a **una prueba de sensibilidad**: por ejemplo, recalcular tasas/predicciones excluyendo el evento `Chete1-20` con fecha imputada y discutir si cambia la conclusión. El bootstrap ya satisface la exigencia de cuantificar incertidumbre. |
| **Equipo: predicción** | 15 | **Avanzado; pendiente conclusión** | Ya tienen dos modelos, MAE/WAPE, validación cronológica e IP para A. **Elegir el modelo operativo**, comparar formalmente con la **estimación de Fermi revisada**, explicar la utilidad y el margen de error. Si eligen B como modelo final, su incertidumbre predictiva también debe justificarse. |
| **Equipo: pregunta propia (punto 4)** | 5 | **Parcial** | Presentar la pregunta de minería/depósitos y una respuesta limitada por la evidencia: diferencias de capturas **sí**, correlación con minería **no evaluable** con las variables actuales. Especificar qué medición faltaría. Opcional: comprobar si hay coordenadas diferentes para Chete1 y Chete2. |
| **Equipo: presentación y reproducibilidad** | 8 | **Por verificar** | Asegurar notebook ejecutable **de principio a fin** con acceso documentado al CSV; preparar figuras legibles, presentación y narración de expectativa → hallazgos → predicción → decisión que cambió. |
| **Individual: comprensión y sustentación** | 20 | **Por realizar** | Cada integrante debe explicar agrupación, normalización, Pearson, bootstrap, regresión binomial negativa, MAE/WAPE, validación temporal e IC vs. IP; y justificar límites de inferencia. |
| **Individual: contribución** | 10 | **Por verificar** | Guardar aportes visibles por integrante en el repositorio (commits, archivos, comentarios y decisiones). |
| **Individual: reflexión** | 10 | **Por redactar/verificar** | Explicar qué supuesto de Fermi cambió, qué dato o resultado provocó el cambio y qué propuestas hizo la IA frente a cuáles decisiones tomó el equipo. |
| **Total** | **100** | | |

## 3.1. Entregables que exige expresamente el PDF

- [ ] `EstimacionFermi_00/`: estimación **original**, cifra, **rango**, supuestos y supuesto dominante (máximo **2 páginas**); evidencia de su entrega inicial, si existe.
- [ ] `PlanAnalisis_01/`: `plan_analisis.md` con decisiones y cambios, **hoja de campo** y descripción de las campañas reservadas para evaluación.
- [ ] **Síntesis del plan**, máximo **2 páginas**. El PDF utiliza dos variantes del nombre: `plan_analisis_sintesis.md` y `sintesis_plan_analisis.md`; conviene escoger una y aclararlo en el README.
- [ ] `SintesisFinal_02/`: notebook **reproducible**, **presentación** y resultados para la **sustentación**.
- [ ] Evidencia de contribuciones y reflexión individual sobre el uso de IA.

**Advertencia sobre la reserva de campañas:** el PDF pide definirla *antes* de explorar los datos. Dado que el notebook ya contiene exploración y validación sobre las campañas históricas, no se puede reconstruir después una reserva verdaderamente ciega. Deben explicarlo con transparencia y llamar al procedimiento existente **validación retrospectiva temporal**. Solo datos no consultados previamente podrían constituir una evaluación final intacta.

---

# 4. Orden mínimo recomendado para cerrar el trabajo

1. **Verificar la ejecución completa del notebook** desde una sesión limpia y documentar cómo obtener/cargar `occurrences.csv`. El notebook usa `files.upload()`, por lo que requiere el archivo como entrada.
2. **Hacer una sola prueba de sensibilidad** sobre la fecha imputada o sobre eventos con esfuerzo inconsistente; reportar si el hallazgo elegido se mantiene.
3. **Cerrar la comparación de modelos:** describir su mejora pequeña, indicar cuál se utilizará para planear recursos y explicar el significado operativo del IP de **[132; 782]**. No hace falta una red neuronal ni otro algoritmo.
4. **Revisar Fermi:** completar el rango y el supuesto dominante; contrastar con el modelo **sin equiparar diseños de captura distintos**.
5. **Resolver el punto 4:** formular la hipótesis ambiental, usar las diferencias entre sitios como evidencia descriptiva y reconocer que falta la medición de minería/depósitos para calcular una correlación.
6. **Completar documentos y repositorio:** hoja de campo, plan, síntesis de dos páginas, bitácora actualizada, contribuciones individuales y reflexión sobre IA.
7. **Preparar la presentación y sustentación:** resultados verificables, limitaciones y decisión práctica para una próxima campaña.

## 4.1. Qué NO hace falta agregar

No se requiere PCA, random forest, SVM, clasificación, redes neuronales ni una búsqueda extensa de modelos. El PDF prioriza **decisiones justificadas, una predicción útil y evaluación honesta**, no la acumulación de técnicas.

---

# 5. Conclusión para el equipo

La parte computacional principal está desarrollada: limpieza, tasas normalizadas, exploración mensual, correlaciones, dos modelos predictivos, métricas e incertidumbre. **Para cerrar la rúbrica, el trabajo pendiente es principalmente metodológico, documental y de interpretación.** La mejora estadística imprescindible sería **una prueba de sensibilidad sencilla** del hallazgo elegido; la comprobación de los supuestos y convergencia del ajuste binomial negativo también aportaría rigor.

Respecto de la hipótesis sobre agua y minería, el hallazgo defendible es la **heterogeneidad de capturas entre sitios**, no una correlación demostrada con actividad minera. La respuesta científicamente correcta debe distinguir expresamente ambas afirmaciones.

> **Nota de autoría:** las elecciones de localidad, preguntas temporales, uso del esfuerzo y motivación por los depósitos de agua se documentan como parte de las decisiones del equipo; los modelos A y B y el bootstrap fueron propuestas metodológicas discutidas con la IA y posteriormente implementadas por el equipo. Se recomienda completar esta atribución con las contribuciones reales de cada integrante.
