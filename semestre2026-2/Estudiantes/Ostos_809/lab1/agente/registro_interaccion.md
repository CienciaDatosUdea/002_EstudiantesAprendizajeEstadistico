FASE: OBSERVE

Estoy trabajando con el conjunto de datos Palmer Penguins.

Actúa como un agente de análisis exploratorio de datos. Tu responsabilidad es proponer un plan de observación inicial del conjunto de datos.

Restricciones:

- No ejecutes código.
- No calcules estadísticas.
- No inventes resultados.
- En esta fase todavía no hay artifacts disponibles.
- Propón únicamente las operaciones necesarias para conocer la estructura y calidad del conjunto de datos.

Entrega:

1. Un plan de observación.
2. Las variables que consideras necesario inspeccionar.
3. Las operaciones que debería implementar el Runner.
4. Los artifacts que deberían generarse.
Recuerda que el Runner será el encargado de ejecutar todos los cálculos.

Respuesta:
1. Plan de observación
Cargar el CSV y confirmar que se leyó correctamente.
Inspeccionar su estructura: número de filas y columnas, nombres de columnas y tipos de datos.
Revisar la calidad básica: valores faltantes, filas duplicadas y cardinalidad por columna.
Distinguir variables categóricas y numéricas para detectar tipos inesperados. No calcular distribuciones ni estadísticas descriptivas en esta fase.

2. Variables a inspeccionar
Confirmar las columnas presentes en la versión cargada y revisar:

Categóricas: species, island, sex
Numéricas: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
Para cada variable: tipo de dato, faltantes y cantidad de valores distintos. En las categóricas, también conviene registrar los valores observados para detectar etiquetas inconsistentes, sin interpretarlos todavía.

3. Operaciones para el Runner
El Runner debería ejecutar la lectura del CSV y generar un perfil estructural con:

-Dimensiones y nombres de columnas.
-Tipo de dato de cada columna.
-Cantidad de valores faltantes por columna.
-Cantidad de filas duplicadas.
-Cardinalidad por columna.
-Valores distintos de las variables categóricas, para revisar posibles inconsistencias.

El esquema debe reflejar las columnas realmente cargadas, no asumir que coinciden con la lista esperada.

4. Artifacts propuestos
perfil.json: perfil estructural y controles básicos de calidad, siguiendo el formato previsto en runner.py.
No se generan artifacts en esta respuesta; se propone el archivo para cuando el Runner ejecute el plan.

FASE: DESCRIBE 
Actúa como un agente especializado en análisis exploratorio de datos.

contexto disponible, Se encuentra disponible el siguiente artifact:

artifacts/perfil.json

Este archivo contiene información sobre: 
- Número de filas y columnas.
- Nombres de las variables.
- Tipos de datos.
- Valores faltantes.
- Registros duplicados.
- Cardinalidad de las variables

Debes utilizar exclusivamente la información contenida en este artifact para contextualizar tus recomendaciones. el objetivo es Proponer un plan de análisis descriptivo y visualización exploratoria que permita comprender las características y relaciones presentes en el conjunto de datos. la idea es recibir Un resumen del perfil del dataset disponible con un plan de operaciones descriptivas y demás.

No ejecutes código Python y No calcules estadísticas ni inventes resultados, No afirmes correlaciones, diferencias o patrones que todavía no hayan sido comprobados, El Runner será el encargado de ejecutar todos los cálculos y generar las gráficas. 

Respuesta:
Resumen del perfil disponible
El artifact perfil.json describe 344 filas y 7 variables. species, island y sex tienen tipo object; las otras cuatro variables tienen tipo float64.

- Variables categóricas: species (3 valores distintos), island (3) y sex (2).
- Variables numéricas: bill_length_mm (164 valores distintos), bill_depth_mm (80), flipper_length_mm (55) y body_mass_g (94).
- Faltantes: 0 en species e island; 2 en cada variable numérica; 11 en sex.
Filas duplicadas: 0.

Plan de análisis descriptivo y visualización

Revisar la completitud: resumir los faltantes por variable y mostrar visualmente dónde se concentran. En los análisis posteriores, indicar cuántas observaciones válidas se usan; no asumir una estrategia de imputación.
Describir variables categóricas: calcular frecuencias y proporciones de species, island y sex; visualizarlas con gráficos de barras.
Describir variables numéricas: calcular resúmenes descriptivos y visualizar distribuciones con histogramas y diagramas de caja. El Runner debe calcularlos; este plan no presupone resultados.
Explorar relaciones entre variables: generar gráficos de dispersión y diagramas de caja para variables numéricas, tanto en conjunto como agrupadas por species, island y sex. Complementar con tablas cruzadas entre las variables categóricas.
Mantener los resultados separados de las conclusiones: las gráficas y tablas permitirán examinar distribuciones y posibles relaciones, pero cualquier patrón o diferencia deberá verificarse con los resultados del Runner.

Salidas propuestas para el Runner

- Tablas de frecuencias categóricas y resúmenes descriptivos numéricos.
- Resumen visual de faltantes.
- - Gráficos de barras, histogramas, diagramas de caja y dispersión.
Tablas cruzadas y, si resulta útil para comparar variables numéricas, una matriz de correlaciones calculada sobre observaciones válidas.

Las salidas deben incluir los nombres de las variables y el tratamiento de faltantes utilizado, sin presentar relaciones como hallazgos antes de revisar los resultados.

FASE: HYPOTHESIZE_AND_CONCLUDE 

Identificar preguntas de investigación relevantes a partir del análisis exploratorio, formular hipótesis estadísticas y proponer pruebas que permitan evaluar las relaciones o diferencias observadas.

## Restricciones

1. No ejecutes código Python.
2. No inventes resultados ni valores estadísticos.
3. No afirmes que una hipótesis fue aceptada o rechazada sin contar con los resultados de las pruebas.
4. Justifica la selección de cada prueba estadística.
5. Considera los valores faltantes y los supuestos de las pruebas.
6. El Runner será el encargado de ejecutar los cálculos estadísticos.

## Actividades
**A. Formulación de hipótesis**

Propón preguntas de investigación que puedan responderse con las variables disponibles.

Para cada pregunta, especifica:

- Pregunta de investigación.
- Hipótesis nula (H0).
- Hipótesis alternativa (H1).
- Variables involucradas.
- Prueba estadística recomendada.
- Justificación de la prueba.
- Supuestos que deben verificarse.
**B. Ejecución estadística**

Indica al Runner qué operaciones debe ejecutar para evaluar cada hipótesis.

Considera, cuando corresponda:

- Correlación de Pearson o Spearman.
- ANOVA o una alternativa no paramétrica.
- Prueba chi-cuadrado de independencia.
- Pruebas post hoc cuando sean necesarias.
**C. Interpretación**

Una vez disponibles los resultados del Runner, interpreta los estadísticos y valores p. Distingue entre significancia estadística y relevancia práctica. No confundas asociación con causalidad.

## Artifacts solicitados
Propón la estructura de los siguientes archivos:

- `06_hypotheses_log.json`: preguntas, hipótesis y pruebas propuestas.
- `08_tests.json`: resultados de las pruebas estadísticas.
- `07_conclusions.json`: interpretación y conclusiones respaldadas por los resultados.

## Formato de respuesta
Entrega:

1. Preguntas de investigación identificadas.
2. Hipótesis nulas y alternativas.
3. Pruebas estadísticas recomendadas y su justificación.
4. Operaciones específicas que debe ejecutar el Runner.
5. Estructura de los artifacts que se deben generar.
6. Advertencias sobre supuestos y limitaciones.
Recuerda: primero propones las hipótesis y las pruebas; el Runner genera los resultados. Solo después podrás formular conclusiones.

Respuesta:
## Perfil disponible

Según `perfil.json`, el dataset contiene 344 filas y 7 variables: tres categóricas (`species`, `island`, `sex`) y cuatro numéricas (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`). Hay 2 valores faltantes en cada variable numérica, 11 en `sex` y ninguno en `species` e `island`. El perfil registra 0 filas duplicadas.

Las preguntas siguientes son hipótesis para evaluar, no resultados. No se dispone aquí de salidas de pruebas, así que no corresponde concluir si alguna se sostiene.

## 1–3. Preguntas, hipótesis y pruebas recomendadas

| ID | Pregunta e hipótesis | Variables | Prueba recomendada y justificación |
|---|---|---|---|
| H1 | ¿Difiere `body_mass_g` entre las categorías de `species`? **H0:** las medias son iguales en todas las categorías. **H1:** al menos una media difiere. | `species`, `body_mass_g` | ANOVA de un factor si los residuos son adecuados y las varianzas son homogéneas; ANOVA de Welch si las varianzas difieren. Kruskal–Wallis si los supuestos paramétricos no son razonables. Compara una variable numérica entre más de dos grupos. |
| H2 | ¿Difiere `flipper_length_mm` entre las categorías de `species`? **H0:** las medias son iguales en todas las categorías. **H1:** al menos una media difiere. | `species`, `flipper_length_mm` | Mismo criterio que H1; compara una respuesta numérica entre más de dos grupos. |
| H3 | ¿Se asocian `bill_length_mm` y `body_mass_g`? **H0:** la correlación poblacional es cero. **H1:** es distinta de cero. | `bill_length_mm`, `body_mass_g` | Pearson si la relación es aproximadamente lineal y no está dominada por valores atípicos; Spearman si la relación es monotónica pero no lineal, o si los supuestos de Pearson no son adecuados. |
| H4 | ¿Son independientes `species` e `island`? **H0:** son independientes. **H1:** existe asociación. | `species`, `island` | Chi-cuadrado de independencia: ambas variables son categóricas. Revisar los conteos esperados; si son escasos, considerar una prueba exacta o una aproximación por Monte Carlo. |
| H5 | ¿Difiere `body_mass_g` entre las dos categorías observadas de `sex`? **H0:** las medias son iguales. **H1:** las medias difieren. | `sex`, `body_mass_g` | Prueba *t* de Welch para grupos independientes, adecuada sin asumir varianzas iguales; Mann–Whitney como alternativa si los datos presentan distribuciones muy asimétricas o valores atípicos que comprometen el análisis paramétrico. |

Para H1 y H2, si la prueba global detecta evidencia de diferencias, el Runner puede efectuar comparaciones post hoc: Tukey tras ANOVA estándar, Games–Howell tras Welch o Dunn con ajuste por multiplicidad tras Kruskal–Wallis. El ajuste debe corresponder a la prueba elegida.

## 4. Operaciones para el Runner

1. Para cada hipótesis, seleccionar las filas completas únicamente para las variables de esa prueba y registrar el número de casos incluidos y excluidos. El perfil no indica cuántas filas tienen faltantes simultáneamente en las variables de una misma prueba.
2. Examinar los supuestos relevantes antes de elegir la prueba: independencia de observaciones, forma de las distribuciones o residuos, valores atípicos, homogeneidad de varianzas y, para chi-cuadrado, conteos esperados.
3. Ejecutar cada prueba con hipótesis bilateral salvo que se justifique y preregistre una alternativa direccional.
4. Si corresponde, ejecutar las pruebas post hoc y ajustar los valores p por comparaciones múltiples.
5. Reportar estadístico, grados de libertad cuando apliquen, valor p, tamaño del efecto e intervalo de confianza. Para asociaciones categóricas, incluir una medida como V de Cramér; para comparaciones de grupos, una medida de efecto compatible con la prueba.
6. No convertir una asociación en una afirmación causal. Interpretar conjuntamente incertidumbre, tamaño del efecto y relevancia práctica; un valor p no mide por sí solo la importancia de una diferencia.

## 5. Estructura propuesta de artifacts

Los ejemplos son **esquemas**: los campos de resultados quedan pendientes hasta que el Runner ejecute las pruebas.

`06_hypotheses_log.json`:

```json
{
  "status": "proposed",
  "hypotheses": [
    {
      "id": "H1",
      "question": "¿Difiere body_mass_g entre las categorías de species?",
      "null_hypothesis": "Las medias de body_mass_g son iguales entre las categorías de species.",
      "alternative_hypothesis": "Al menos una media de body_mass_g difiere.",
      "variables": ["species", "body_mass_g"],
      "recommended_test": "ANOVA; Welch o Kruskal-Wallis según los diagnósticos",
      "justification": "Comparar una variable numérica entre más de dos grupos.",
      "assumptions_to_check": [
        "Independencia de observaciones",
        "Distribución de residuos y valores atípicos",
        "Homogeneidad de varianzas para ANOVA estándar"
      ],
      "post_hoc": "Solo si procede según la prueba global; método y ajuste dependientes de la prueba"
    }
  ]
}
```

Registrar H2–H5 con los mismos campos.

`08_tests.json`:

```json
{
  "status": "pending",
  "results": [
    {
      "hypothesis_id": "H1",
      "test_used": null,
      "n_included": null,
      "n_excluded": null,
      "assumption_checks": {},
      "statistic": null,
      "degrees_of_freedom": null,
      "p_value": null,
      "effect_size": null,
      "confidence_interval": null,
      "post_hoc_results": [],
      "decision": null
    }
  ]
}
```

`07_conclusions.json`:

```json
{
  "status": "pending_results",
  "conclusions": [
    {
      "hypothesis_id": "H1",
      "evidence_summary": null,
      "statistical_significance": null,
      "practical_relevance": null,
      "limitations": [],
      "causal_claim": false
    }
  ]
}
```

## 6. Advertencias

La prueba final depende de los diagnósticos y del diseño de muestreo; el perfil estructural no permite confirmar independencia, normalidad ni otros supuestos. Deben informarse los datos excluidos por faltantes y evitar imputarlos sin una decisión metodológica explícita. Si se hacen varias pruebas o comparaciones post hoc, hay que controlar la multiplicidad. Las conclusiones de `07_conclusions.json` deben redactarse únicamente después de recibir `08_tests.json` completo.