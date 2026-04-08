# Observatorio de Datos — Informe Final
## Dataset: EUR/USD Historical OHLC (1H)
**Lab:** Lab_02_pandas | **Estudiante:** Moscoso_80 | **Fecha:** 2026-03-26

---

## 1. Descripción del Dataset

El dataset contiene 5.000 observaciones horarias del par de divisas EUR/USD correspondientes al período 2022-07-25 13:00:00 – 2023-05-12 23:00:00 (~10 meses), descritas por 13 columnas: un identificador secuencial (`id`), una marca temporal (`time`), cuatro precios de vela OHLC (`open`, `high`, `low`, `close`), dos métricas de liquidez y costo transaccional (`tick_volume`, `spread`, `real_volume`), y cuatro variables derivadas calculadas por el broker (`MeanCloseOpen`, `Diff_Close`, `Diff_Open`, `Diff_MeanCloseOpen`). [`runner output: Shape 5000 rows × 13 columns`]

---

## 2. Calidad de los Datos (Fase OBSERVE)

| Verificación | Resultado |
|---|---|
| Valores faltantes | 0 en 5.000 × 13 celdas |
| Filas duplicadas | 0 |
| Tipo de `time` | `str` — requiere conversión a `datetime` |
| Valores únicos en `real_volume` | 1 (columna constante) |

`[02_missing_report__OBSERVE, 03_duplicates_report__OBSERVE, 01_schema__OBSERVE]`

**Columna descartada:** `real_volume` presenta un único valor en las 5.000 filas — no aporta varianza ni capacidad explicativa y debe excluirse de todo análisis posterior. Su presencia sugiere que el broker censuró el dato o que no aplica para este instrumento. `[student_log.md: Phase_1_Q2]`

**Conversión pendiente:** La columna `time` fue detectada como `str`. Para cualquier análisis temporal (periodicidad, gaps, hora del día) debe convertirse con `pd.to_datetime(df['time'])` antes de continuar. `[student_log.md: Phase_1_Q3]`

**Gaps temporales no detectados:** La ausencia de missing values no garantiza continuidad en la serie. Los cierres de mercado (fines de semana, festivos) producen saltos en la secuencia temporal que el reporte de missing estándar no captura. Este análisis requiere verificación explícita sobre la columna `time` una vez convertida. `[student_log.md: Phase_1_Q4]`

---

## 3. Variables Derivadas

El dataset incluye cuatro columnas calculadas cuya interpretación es relevante para el análisis:

| Columna | Interpretación inferida |
|---|---|
| `MeanCloseOpen` | Promedio o valor medio entre `close` y `open` de cada vela |
| `Diff_Close` | Diferencia del precio de cierre respecto al período anterior (retorno absoluto) |
| `Diff_Open` | Diferencia del precio de apertura respecto al período anterior |
| `Diff_MeanCloseOpen` | Diferencia de `MeanCloseOpen` respecto al período anterior |

La correlación entre `MeanCloseOpen` y las variables OHLC ≈ 1.00 confirma que es matemáticamente derivada de los mismos precios. `[heatmap_corr_pearson__DESCRIBE__20260326_224952.png: MeanCloseOpen-close = 1.00]`

---

## 4. Hallazgos Descriptivos Clave (Fase DESCRIBE)

### 4.1 Redundancia Extrema entre Variables de Precio

Las variables `open`, `high`, `low`, `close` y `MeanCloseOpen` presentan correlaciones de Pearson ≈ 1.00 entre sí en todos los pares. Esto no implica que sean idénticas vela a vela, sino que comparten la misma **tendencia temporal** durante el período: al haber un movimiento alcista sostenido, todos los niveles de precio crecen juntos a lo largo de las 5.000 filas.

`[heatmap_corr_pearson__DESCRIBE__20260326_224952.png: bloque OHLC-MeanCloseOpen ≈ 1.00]`

Usar todas como features independientes en un modelo introduciría multicolinealidad severa. El estudiante identificó correctamente este problema. `[student_log.md: Phase_2_Q1]`

### 4.2 Tendencia Temporal en el Nivel de Precio

La variable `id` (índice secuencial que sirve como proxy de tiempo) presenta correlación de Pearson ≈ 0.86 con todas las variables de precio. Esta es la firma estadística de una **serie no estacionaria con tendencia**: el nivel medio del EUR/USD en este período fue creciente.

`[heatmap_corr_pearson__DESCRIBE__20260326_224952.png: id-close = 0.86, id-open = 0.86]`

El estudiante inicialmente describió esta correlación como "mera casualidad", pero el scatter `id vs high` confirmó visualmente una tendencia alcista neta con fluctuaciones importantes — especialmente una depresión pronunciada alrededor de id ≈ 1.100-1.200 (precio ≈ 0.96) seguida de una recuperación sostenida hasta ≈ 1.10. `[scatter_id_vs_high__DESCRIBE__20260326_224952.png, student_log.md: Phase_3_Q2]`

### 4.3 Independencia entre Variables de Nivel y de Diferencia

Las variables `Diff_Close`, `Diff_Open` y `Diff_MeanCloseOpen` presentan correlación ≈ 0.00 con las variables de nivel (`open`, `high`, `low`, `close`), pero se correlacionan con `Diff_MeanCloseOpen` al nivel de 0.70 entre sí.

`[heatmap_corr_pearson__DESCRIBE__20260326_224952.png: Diff_Close-close ≈ 0.00, Diff_Close-Diff_MeanCloseOpen = 0.70, Diff_Open-Diff_MeanCloseOpen = 0.70]`

Esto es la manifestación empírica del concepto de **estacionariedad**: al diferenciar una serie con tendencia, se elimina la componente de nivel y se obtiene una serie cuya media y varianza no dependen del tiempo. Las variables `Diff_*` capturan el cambio período a período (retornos absolutos), que son aproximadamente estacionarias, mientras que los niveles no lo son. El estudiante no conocía el término formal pero intuyó correctamente la independencia entre ambos tipos. `[student_log.md: Phase_2_Q3, Phase_3_Q3]`

### 4.4 Relación entre Tick Volume y Precio

`tick_volume` presenta correlación de Pearson entre -0.28 y -0.31 con todas las variables de precio, y -0.23 con `id`.

`[heatmap_corr_pearson__DESCRIBE__20260326_224952.png: tick_volume-close = -0.30, tick_volume-id = -0.23]`

Sin embargo, el scatter `close vs tick_volume` revela una nube completamente dispersa sin patrón lineal identificable visualmente. La correlación negativa moderada existe en el cómputo global, pero queda enmascarada por la enorme varianza del volumen en cada nivel de precio. `[scatter_close_vs_tick_volume__HYPOTHESIZE__20260326_234412.png, student_log.md: Phase_3 observación plots]`

### 4.5 Distribución del Spread

El histograma de `spread` muestra una distribución fuertemente concentrada en valores bajos (primer bin, mayoría de las 5.000 observaciones), con un cluster secundario visible alrededor de 25–40 y outliers aislados que alcanzan ≈ 175.

`[hist_spread__DESCRIBE__20260326_224951.png]`

El scatter `id vs spread` reveló que los eventos de spread alto (> 25) **no están concentrados en períodos temporales específicos**: aparecen de forma aproximadamente uniforme a lo largo de todo el rango temporal, refutando la hipótesis H3. `[scatter_id_vs_spread__HYPOTHESIZE__20260326_234412.png, student_log.md: Phase_3 observación plots]`

---

## 5. Pre-análisis del Estudiante vs. Hallazgos

| Predicción del estudiante (Phase 0) | Hallazgo | Resultado |
|---|---|---|
| Correlación entre `open` y `close` sería inexistente o débil (el mercado fluctúa) | r(open, close) ≈ 1.00 debido a tendencia temporal compartida | Refutado — la tendencia explica la correlación |
| Esperaba comportamiento aleatorio en general | id-close ≈ 0.86; scatter id vs high muestra tendencia alcista neta | Parcialmente refutado |
| Predijo posible tendencia **negativa** (auge de la IA favorecería USD) | La tendencia fue **positiva** — EUR/USD subió de ≈0.96 a ≈1.10 | Dirección incorrecta |
| Las variables derivadas (`Diff_*`) se relacionarían con `tick_volume` | Diff_* tienen correlación ≈ 0.00 con tick_volume; es el *nivel* el que difiere negativamente | Parcialmente incorrecto |
| `real_volume` podría ser útil para análisis | Columna constante — descartada | Correctamente sospechado como "censurada" |
| Los huecos temporales no serían detectados por el reporte de missing | Confirmado — requiere análisis explícito sobre `time` convertida | Correcto |

---

## 6. Patrones Visuales

**scatter_id_vs_high** `[scatter_id_vs_high__DESCRIBE__20260326_224952.png]`
Confirma visualmente la tendencia alcista neta del EUR/USD en el período. Se observa una depresión pronunciada en id ≈ 1.100–1.200 (precio ≈ 0.96) y una recuperación sostenida hasta ≈ 1.10. El crecimiento no es uniforme — hay ciclos de subida y bajada sobre la tendencia — pero la dirección general es positiva. El estudiante señaló correctamente que el intervalo temporal (~10 meses) es insuficiente para identificar "seasons" estadísticamente robustas.

**hist_spread** `[hist_spread__DESCRIBE__20260326_224951.png]`
Distribución bimodal asimétrica: modo dominante en valores bajos (spread normal de mercado) y cluster secundario en 25–40. La interpretación del estudiante sobre dos regímenes transaccionales del broker es plausible, aunque el scatter temporal descartó la concentración periódica.

**scatter_id_vs_spread** `[scatter_id_vs_spread__HYPOTHESIZE__20260326_234412.png]`
Los eventos de spread alto (>25) aparecen distribuidos de forma aproximadamente uniforme a lo largo de todo el período. No se observa concentración en fechas específicas. Esto refuta la hipótesis H3 en su formulación temporal.

**scatter_close_vs_tick_volume** `[scatter_close_vs_tick_volume__HYPOTHESIZE__20260326_234412.png]`
Nube de alta dispersión sin patrón lineal visualmente identificable. El volumen presenta varianza extrema (valores entre ~0 y >20.000) en todos los niveles de precio. La correlación negativa global (r ≈ -0.30) existe pero está completamente enmascarada por el ruido.

---

## 7. Hipótesis y Estado de Evaluación

| ID | Origen | Enunciado resumido | Estado | Evidencia clave |
|---|---|---|---|---|
| H1 | Estudiante | EUR/USD tuvo tendencia alcista sostenida en jul 2022 – may 2023 | **Confirmado visualmente** | scatter id vs high muestra pendiente positiva neta con dip intermedio |
| H2 | Estudiante | A mayor nivel de precio, menor tick_volume | **Inconcluso** | r=-0.30 en heatmap, pero scatter visual muestra alta dispersión sin patrón claro |
| H3 | Estudiante | Eventos de spread alto concentrados en períodos específicos | **Refutado** | scatter id vs spread muestra distribución uniforme en el tiempo |
| H4 | AI Observer | Las variables Diff_* son estacionarias; los niveles no | **Plausible, sin test formal** | Diferencia de correlación (nivel-nivel ≈ 1.00 vs. nivel-diff ≈ 0.00) es consistente con la hipótesis |

`[06_hypotheses_log__HYPOTHESIZE__20260326, student_log.md: Phase_3]`

---

## 8. Qué Probar a Continuación

1. **H1 — Confirmar tendencia con test formal:** Aplicar Mann-Kendall trend test sobre la serie `close` ordenada por `id`. Complementar con regresión lineal `close ~ id` y evaluar significancia de la pendiente.

2. **H2 — Explorar la relación volumen-precio con mayor resolución:** El scatter muestra alta heteroscedasticidad. Aplicar correlación de Spearman (más robusta a distribuciones no normales) y segmentar el análisis por cuartiles de precio para verificar si la asociación negativa persiste en cada segmento por separado.

3. **H3 — Re-explorar spread por hora del día:** El análisis temporal con `id` no detectó concentración. Una vez convertida `time` a `datetime`, analizar la distribución del spread por hora del día (sesiones de Londres, Nueva York, Asia) y por día de la semana. Es posible que la concentración exista a escala intradiaria y no sea visible a escala mensual.

4. **H4 — Test de estacionariedad:** Aplicar Augmented Dickey-Fuller (ADF) sobre `close` y sobre `Diff_Close`. La hipótesis predice que `close` no rechaza la hipótesis nula de raíz unitaria, mientras que `Diff_Close` sí la rechaza. Este es un paso estándar en el análisis de series de tiempo financieras antes de cualquier modelado.

5. **Selección de features para modelado:** Dado el bloque de correlación ≈ 1.00 en OHLC, usar solo `close` (o `MeanCloseOpen`) como representante del nivel de precio. Incluir `tick_volume`, `spread` (con tratamiento especial para outliers > 25), y las variables `Diff_*` como features de cambio. Excluir `real_volume`, `id` y las demás variables de precio redundantes.

---

*Artefactos fuente:*
`00–03_*.json (OBSERVE)` · `04a/b/d_*.json (DESCRIBE)` · `05_visual_registry*.json (DESCRIBE)` · `heatmap_corr_pearson__DESCRIBE__20260326_224952.png` · `hist_spread__DESCRIBE__20260326_224951.png` · `scatter_id_vs_high__DESCRIBE__20260326_224952.png` · `scatter_id_vs_spread__HYPOTHESIZE__20260326_234412.png` · `scatter_close_vs_tick_volume__HYPOTHESIZE__20260326_234412.png` · `06_hypotheses_log__HYPOTHESIZE__20260326.json` · `07_conclusions__HYPOTHESIZE__20260326.json` · `student_log.md`
