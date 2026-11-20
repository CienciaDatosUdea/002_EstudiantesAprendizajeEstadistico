# Reto Final: Síntesis del Plan de Análisis

**Curso:** Aprendizaje Estadístico 2026-2  
**Estudiantes:** Jeronimo Lopez Gomez - Jose Daniel Godoy

---

## Pregunta y estimación inicial

**¿Cuántos individuos del género *Anopheles* podrían capturarse en una próxima campaña de cuatro noches en Cheté, y con qué incertidumbre?**

Nuestra estimación de Fermi, realizada sin consultar los datos históricos, fue de **532 mosquitos en total**, de los cuales **346 serían *Anopheles***. Consideramos escenarios de **122 a 605 *Anopheles***; estos valores son supuestos hipotéticos, **no intervalos de confianza**.

## Tres decisiones fundamentales

- **Observación:** partimos de un registro de captura con información de fecha, lugar, especie y cantidad de individuos. Para comparar capturas, agruparemos por jornada o campaña y tendremos en cuenta el esfuerzo de muestreo disponible. Un registro de ocurrencia no siempre representa un solo mosquito.
- **Qué predeciremos:** el número de *Anopheles* en cuatro noches en Cheté bajo condiciones de muestreo comparables. Si los registros no permiten reconstruir campañas equivalentes, informaremos esa limitación y consideraremos una estimación por jornada o por unidad de esfuerzo.
- **Cómo sabremos si funciona:** compararemos los valores estimados con los observados mediante **MAE** (y, de ser útil, RMSE); contrastaremos con el promedio histórico y con la estimación de Fermi revisada. Reservaremos las campañas más recientes para una evaluación final, respetando el orden temporal y manteniendo agrupadas las observaciones relacionadas.

## Fuentes y alcance

Trabajaremos principalmente con los registros Darwin Core del INS (`occurrence.txt`) para Cheté. Usaremos otras localidades de Timbiquí para comparaciones exploratorias, sin mezclarlas automáticamente. Complementaremos con precipitación de DHIME-IDEAM (`dhime_mensual.csv` y `dhime_diario.csv`) y con nuestra propuesta de hoja de campo (`hoja_de_campo.md`).

## Preguntas principales del EDA

1. ¿Cómo se distribuyen los registros por fecha, localidad y especie?
2. ¿Qué calidad tienen los datos: fechas, coordenadas, duplicados, conteos y esfuerzo de muestreo?
3. ¿Cómo varían las capturas entre meses y jornadas, considerando el esfuerzo disponible?
4. ¿Qué patrones aparecen al comparar la precipitación y las capturas?
5. ¿Cómo se comparan las agrupaciones exploratorias de cuatro jornadas en Cheté con los **346 *Anopheles*** previstos en Fermi?

## Herramientas propuestas

Usaremos tablas, gráficas y medidas de dispersión en el EDA; **bootstrap** para expresar incertidumbre, siempre agrupando unidades de muestreo comparables; y, si la cantidad y calidad de campañas lo permiten, **una regresión sencilla** para la predicción. Las variables deberán estar disponibles **antes** de la campaña que se busca predecir.

## Alcances y precauciones

- Las agrupaciones de cuatro fechas de `EDA.ipynb` **no son necesariamente campañas reales consecutivas**.
- Debemos verificar qué significa `individualCount` y no asumir que un valor faltante equivale a un mosquito.
- Un día registrado no equivale automáticamente a una jornada completa ni a una cantidad fija de horas-colector.
- Los registros no permiten, por sí solos, determinar diferencias interior/exterior ni explicar características particulares de las viviendas si esas variables no están documentadas.
- Las precipitaciones de estaciones externas pueden diferir de la lluvia efectiva en Cheté; una asociación con capturas no demuestra causalidad.

**Producto esperado:** una comparación razonada entre Fermi y los datos históricos, una estimación acompañada de incertidumbre y, si se implementa el modelo, una evaluación temporal honesta. Registraremos cualquier cambio de decisiones en el plan completo.
