# Reto Final: Síntesis del Plan de Análisis

**Curso:** Aprendizaje Estadístico 2026-2  
**Estudiantes:** Jeronimo Lopez Gomez - Jose Daniel Godoy

---

## Pregunta y estimación inicial

**¿Cuántos individuos del género *Anopheles* podrían capturarse en una próxima campaña de cuatro noches en Cheté, y con qué incertidumbre?**

Nuestra estimación de Fermi, realizada sin consultar los datos históricos, fue de **532 mosquitos en total**, de los cuales **346 serían *Anopheles***. Consideramos escenarios de **122 a 605 *Anopheles***; estos valores son supuestos hipotéticos, **no intervalos de confianza**.

## Tres decisiones fundamentales

- **Observación:** cada registro es una especie en una casa durante una campaña mensual y acumula varias noches. Usamos la **visita** (casa × mes) y, para el modelo, la **campaña** de Cheté (sus dos casas sumadas). La captura se mide como ***Anopheles* por casa-hora** (individuos / [noches × horas]), suponiendo un colector por casa.
- **Qué predeciremos:** el número de *Anopheles* en cuatro noches en Cheté con nuestro diseño (5 colectores × 5 h × 4 noches = 100 colector-horas). Usaremos una **regresión lineal sin intercepto**, $\hat N = \beta \times \text{colector-horas}$, basada en **H1: la captura es proporcional al esfuerzo**. Es la estimación de Fermi con la tasa calibrada con datos.
- **Cómo sabremos si funciona:** mediremos el **MAE** frente a Fermi original (tasa de 3,46 por colector-hora). Haremos una validación temporal (cada campaña se predice solo con las anteriores) y reservaremos las 4 campañas más recientes para la evaluación final.

## Fuentes y alcance

Trabajaremos principalmente con los registros Darwin Core del INS (`occurrence.txt`) para Cheté, dentro de Guapi y Timbiquí; los registros de Santa Bárbara (Nariño) se descartan. Compararemos Cheté con las localidades vecinas de la ribera del río Timbiquí (Coteje, Velásquez y San Miguel del Río) sin mezclarlas automáticamente. Complementaremos con precipitación de DHIME-IDEAM (`dhime_mensual.csv` y `dhime_diario.csv`) y con nuestra propuesta de hoja de campo (`hoja_de_campo.md`). El análisis está en `EDA.ipynb` y el modelo en `model.ipynb`.

## Preguntas principales del EDA

1. ¿Cómo se distribuyen los registros por fecha, localidad y especie?
2. ¿Qué calidad tienen los datos: fechas, coordenadas, duplicados, conteos y esfuerzo de muestreo?
3. ¿Cómo varía la tasa de captura (*Anopheles* por casa-hora) entre meses y campañas?
4. ¿Se correlaciona la lluvia (durante la campaña, 14 y 30 días antes) con la cantidad de mosquitos, el esfuerzo y la tasa de captura?
5. ¿La captura es proporcional al esfuerzo (H1)? ¿Cómo se compara la tasa observada en Cheté con la implícita en los **346 *Anopheles*** de Fermi?
6. ¿Las localidades vecinas de la ribera del río Timbiquí son equivalentes a Cheté en tasa y en especies?

## Herramientas propuestas

Usaremos tablas, gráficas, medidas de dispersión y correlación de Spearman en el EDA. Para expresar la incertidumbre usaremos **bootstrap de campañas completas**: IC 95 % de la tasa y del valor esperado, y rango del 80 % de una campaña nueva. Para la predicción, **una regresión lineal sin intercepto** sobre el esfuerzo, la única variable que se conoce **antes** de la campaña. La precipitación queda fuera del modelo porque su relación con la tasa fue débil.

## Alcances y precauciones

- Descartamos las agrupaciones de cuatro fechas: mezclaban localidades y periodos de varios meses. Las noches de una campaña no siempre son consecutivas.
- Debemos verificar qué significa `individualCount` y no asumir que un valor faltante equivale a un mosquito. Solo hay registros cuando hubo capturas, así que no se ven visitas con cero individuos.
- Un día registrado no equivale automáticamente a una jornada completa ni a una cantidad fija de horas-colector. Suponemos un colector por casa, y la tasa medida con ~12 h por noche se aplica a las 5 h de nuestro diseño.
- Los registros no permiten, por sí solos, determinar diferencias interior/exterior: todas las capturas históricas son dentro de la vivienda. Cheté tiene solo 2 casas y 22 campañas, y las casas difieren mucho entre sí.
- Las precipitaciones de estaciones externas pueden diferir de la lluvia efectiva en Cheté; una asociación con capturas no demuestra causalidad.

**Producto esperado:** una comparación razonada entre Fermi y los datos históricos (por tasa horaria), una estimación de la próxima campaña acompañada de incertidumbre y una evaluación temporal honesta del modelo frente a Fermi original. Los cambios de decisiones quedan registrados en el plan completo (`plan_analisis.md`).
