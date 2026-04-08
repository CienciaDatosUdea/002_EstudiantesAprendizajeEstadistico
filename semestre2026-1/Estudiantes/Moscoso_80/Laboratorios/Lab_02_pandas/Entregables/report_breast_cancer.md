# Observatorio de Datos — Informe Final
## Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
**Lab:** Lab_02_pandas | **Estudiante:** Moscoso_80 | **Fecha:** 2026-03-14

---

## 1. Descripción del Dataset

El dataset WDBC contiene 569 observaciones de biopsias de mama, cada una descrita por 32 atributos: un identificador administrativo (`id`), un diagnóstico binario (`diagnosis`: B=benigno, M=maligno), y 30 características numéricas continuas derivadas de imágenes digitalizadas de aspirados con aguja fina (FNA). Las 30 características se organizan en tres grupos de 10 mediciones base: media (`_mean`), error estándar (`_se`) y valor extremo (`_worst`), donde `_worst` corresponde a la media de los tres valores más grandes de cada medición en la imagen. [`00_raw_profile__OBSERVE__20260314_103022.json`]

---

## 2. Calidad de los Datos (Fase OBSERVE)

| Verificación | Resultado |
|---|---|
| Valores faltantes | 0 en 569 × 32 = 18,208 celdas |
| Filas duplicadas | 0 |
| Filas totales | 569 (matches documentación) |
| Columnas totales | 32 (matches documentación) |

`[02_missing_report__OBSERVE__20260314_103022.json, 03_duplicates_report__OBSERVE__20260314_103022.json]`

**Incidencia detectada y corregida:** El archivo CSV original fue leído sin encabezados explícitos, convirtiendo la primera fila de datos en nombres de columna y perdiendo una observación (568 filas en lugar de 569). Corregido asignando los 32 nombres de columna desde la documentación antes de proceder con la Fase DESCRIBE.

La columna `id` tiene 569 valores únicos y std=125,020,585 — confirmada como identificador administrativo. Fue excluida de todo análisis descriptivo y predictivo. `[01_schema__OBSERVE__20260314_103022.json, 04a_numeric_summary__DESCRIBE__20260314_105559.json:id.std]`

---

## 3. Distribución del Target

La variable `diagnosis` presenta **desbalance moderado**:

| Clase | n | Proporción |
|---|---|---|
| B (Benigno) | 357 | 62.74% |
| M (Maligno) | 212 | 37.26% |

`[04b_categorical_summary__DESCRIBE__20260314_105559.json]`

Este desbalance (~2:1) implica que cualquier clasificador que no lo considere tenderá a sesgarse hacia la clase mayoritaria. Las comparaciones grupo a grupo deben interpretarse teniendo en cuenta el tamaño diferencial de los grupos.

---

## 4. Hallazgos Descriptivos Clave

### 4.1 Estructura de Correlación

El mapa de calor de Pearson revela una **estructura de tres bloques diagonales** correspondientes a `_mean`, `_se` y `_worst`. El bloque cruzado `_mean`/`_worst` es más intenso que el bloque `_mean`/`_se`:

- r(`radius_mean`, `radius_worst`) = **0.970** `[04d_correlation_pearson: radius_mean.radius_worst]`
- r(`radius_mean`, `radius_se`) = **0.679** `[04d_correlation_pearson: radius_mean.radius_se]`
- r(`concave_points_mean`, `concave_points_worst`) = **0.910** `[04d_correlation_pearson]`
- r(`concave_points_mean`, `concave_points_se`) = **0.616** `[04d_correlation_pearson]`

Esto es consistente con que `_worst` es un agregado calculado sobre los mismos valores de los que se deriva `_mean` (media de los 3 valores máximos vs. media de todos los valores). `[student_log.md:Phase_1_Q3, student_log.md:Phase_2_Q4]`

### 4.2 Multicolinealidad Extrema en Características de Tamaño

Las tres características de tamaño correlacionan entre sí de forma casi perfecta:

- r(`radius_mean`, `perimeter_mean`) = **0.998** `[04d_correlation_pearson]`
- r(`radius_mean`, `area_mean`) = **0.987** `[04d_correlation_pearson]`
- r(`perimeter_mean`, `area_mean`) = **0.987** `[04d_correlation_pearson]`

Incluir las tres en un modelo introduce tres pesos para información geométricamente dependiente — una limitación identificada correctamente por el estudiante. `[student_log.md:Phase_3_Q7]`

### 4.3 Asimetría Diferencial en el Grupo _se

Las características de error estándar presentan sesgos notablemente más extremos que sus contrapartes `_mean`:

| Feature | Skewness | Kurtosis |
|---|---|---|
| area_se | 5.447 | 49.209 |
| concavity_se | 5.110 | 48.861 |
| radius_se | 3.089 | 17.687 |
| area_mean | 1.646 | 3.652 |
| radius_mean | 0.942 | 0.846 |

`[04a_numeric_summary__DESCRIBE__20260314_105559.json]`

Esto indica que el error estándar de medición dentro de una imagen presenta distribuciones de cola muy pesada — posiblemente generadas por pocos casos con alta variabilidad intra-imagen.

### 4.4 Correlación Negativa de Dimensión Fractal

`fractal_dimension_mean` presenta correlaciones de Pearson negativas con las características de tamaño:
- r(`fractal_dimension_mean`, `radius_mean`) = **-0.312** `[04d_correlation_pearson]`
- r(`fractal_dimension_mean`, `area_mean`) = **-0.283** `[04d_correlation_pearson]`

Esto contrasta con las demás características de forma, que correlacionan positivamente con el tamaño. La interpretación del estudiante es coherente: la dimensión fractal mide la complejidad del borde en una escala relacionada con la rugosidad — una frontera más regular (valor bajo) puede estar asociada a tumores más grandes y regulares. `[student_log.md:Phase_2_Q5]`

---

## 5. Patrones Visuales

**box_radius_mean_by_diagnosis** `[05_visual_registry__DESCRIBE__20260314_105602.json]`
Los rangos intercuartílicos de M y B no se solapan, con solo unos pocos outliers cruzando entre grupos. El estudiante identificó correctamente que la separación visual no es suficiente por sí sola: los valores extremos en ambos grupos crean una zona de ambigüedad. `[student_log.md:Phase_2_Q3]`

**hist_area_mean_by_diagnosis y hist_area_worst_by_diagnosis** `[05_visual_registry__DESCRIBE__20260314_105602.json]`
El grupo B se concentra en valores bajos con distribución estrecha. El grupo M presenta distribución más amplia con cola derecha pronunciada — consistente con skewness=1.646 para area_mean. `[04a_numeric_summary: area_mean.skewness=1.646]`

**scatter_concavity_mean_vs_concave_points_mean_hue_diagnosis** `[05_visual_registry_hypothesize__HYPOTHESIZE__20260314_112737.json]`
Alta correlación entre ambas variables (r=0.921) `[04d_correlation_pearson: concavity_mean.concave_points_mean=0.921]`. El grupo B se concentra en valores bajos de ambas dimensiones; el grupo M se extiende hacia valores más altos, pero la frontera entre grupos es difusa. Una separación lineal en este espacio 2D sería insuficiente para un clasificador robusto. `[student_log.md:Phase_3_Q8]`

---

## 6. Pre-análisis del Estudiante vs. Hallazgos

> El `student_log.md` no contiene una sección formal de Phase 0. Las reflexiones inician en Phase 1. Esta sección compara las hipótesis del estudiante con los hallazgos de Phase 2 y Phase 3.

| Predicción del estudiante | Hallazgo | Resultado |
|---|---|---|
| `id` no debe usarse como feature (alta cardinalidad) | id.n_unique=569, std=125M — confirmado identificador | Correcto |
| `_mean` y `_worst` estarían estrechamente relacionados | r(radius_mean, radius_worst)=0.970 | Correcto |
| `radius_mean` es el mejor separador visual individual | IQR sin solapamiento en box plot | Confirmado visualmente |
| Usar radius, perimeter y area juntos introduce sesgo por redundancia | r > 0.985 pairwise | Correcto |
| Una línea recta en espacio concavity/concave_points sería insuficiente | Frontera difusa en scatter plot | Confirmado |
| `fractal_dimension` tiene escala inversa a regularidad | r negativo con radius y area | Consistent con evidencia |

---

## 7. Hipótesis Propuestas

| ID | Enunciado resumido | Tests sugeridos |
|---|---|---|
| H1 | `area_mean` difiere entre M y B, con M asociado a valores mayores | Mann-Whitney U, t-test independiente |
| H2 | `concavity_mean` y `concave_points_mean` difieren entre M y B con mayor solapamiento que features de tamaño | Mann-Whitney U por feature |
| H3 | `radius_mean`, `perimeter_mean` y `area_mean` son predictores redundantes (r > 0.985) | VIF, PCA |
| H4 | Dentro de cada familia, `_worst` se asocia más con `_mean` que `_se` (r: 0.970 vs 0.679 en radius) | Comparación pareada de correlaciones, clustering jerárquico |

`[06_hypotheses_log__HYPOTHESIZE__20260314.json]`

---

## 8. Qué Probar a Continuación

1. **Pruebas de grupo (H1, H2):** Ejecutar Mann-Whitney U para `area_mean`, `concavity_mean` y `concave_points_mean` separados por `diagnosis`. Requiere agregar un operador `group_stats` o `hypothesis_test` al runner.

2. **Selección de features (H3):** Elegir un representante por cluster de correlación (una feature de tamaño, una de forma, `fractal_dimension_mean`) y comparar rendimiento clasificatorio frente al conjunto completo de 30 features.

3. **Evaluación de features _se (H4):** Generar box plots de cada feature `_se` vs `diagnosis` y comparar visualmente el grado de separación frente a sus contrapartes `_mean` y `_worst`.

4. **Fase de modelado:** El análisis exploratorio indica separabilidad no trivial con solapamiento residual. Un clasificador con capacidad no lineal moderada (e.g. árbol de decisión, SVM con kernel RBF) sería apropiado como siguiente paso, partiendo del subconjunto de features no redundantes identificado en el punto 2.

---

*Artefactos fuente:*
`00–03_*.json (OBSERVE)` · `04a/b/d_*.json (DESCRIBE)` · `05_visual_registry*.json (DESCRIBE + HYPOTHESIZE)` · `06_hypotheses_log__HYPOTHESIZE__20260314.json` · `07_conclusions__HYPOTHESIZE__20260314.json` · `student_log.md`
