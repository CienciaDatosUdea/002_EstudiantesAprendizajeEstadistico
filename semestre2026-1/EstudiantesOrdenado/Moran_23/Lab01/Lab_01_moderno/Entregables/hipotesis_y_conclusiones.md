# 🧪 Hipótesis y Conclusiones Documentadas
**Laboratorio 01: Análisis Exploratorio del Dataset Penguins**  
**Data Observatory | Fase 3: HYPOTHESIZE and CONCLUDE**  
**Universidad de Antioquia — 2026-1 | Estudiante: Moran_23**

---

## 📋 Tabla de Contenidos

1. [Hipótesis 1: Diferencias Intra-Específicas](#hipótesis-1-diferencias-intra-específicas)
2. [Hipótesis 2: Simpson's Paradox](#hipótesis-2-simpsons-paradox)
3. [Hipótesis 3: Sexo como Confosor](#hipótesis-3-sexo-como-confosor)
4. [Hallazgos Descriptivos](#hallazgos-descriptivos)
5. [Patrones Visuales](#patrones-visuales)
6. [Próximos Pasos y Pruebas Estadísticas](#próximos-pasos-y-pruebas-estadísticas)

---

## Hipótesis 1: Diferencias Intra-Específicas

### 🎯 Enunciado

> **Las tres especies de pingüino difieren significativamente en sus medidas morfométricas (longitud del pico, profundidad del pico, longitud de la aleta y masa corporal). Gentoo es la especie más grande en general; Chinstrap tiene picos relativamente más profundos; Adelie es la más pequeña.**

**ID:** H1  
**Origen:** Observaciones del estudiante y datos visuales  
**Tipo:** Hipótesis de diferencia entre grupos (no causal)

### 📊 Variables de Interés

| Variable | Tipo | Rol |
|----------|------|-----|
| `species` | Categórica (3 niveles) | Variable independiente (predictor) |
| `bill_length_mm` | Numérica | Variable dependiente |
| `bill_depth_mm` | Numérica | Variable dependiente |
| `flipper_length_mm` | Numérica | Variable dependiente |
| `body_mass_g` | Numérica | Variable dependiente |

### 📈 Pruebas Estadísticas Sugeridas

1. **ANOVA de una vía** (para cada variable numérica vs species)
   - Asume normalidad e igualdad de varianzas
   - Null: Las medias de cada variable son iguales entre especies
   - Alt: Al menos una especie difiere significativamente

2. **Kruskal-Wallis** (alternativa no-paramétrica)
   - No requiere normalidad
   - Usa rangos en lugar de valores brutos
   - Más robusto ante outliers

3. **Post-hoc Tukey HSD** (comparaciones pairwise)
   - Determina QUÉ pares de especies difieren
   - Controla tasa de error familywise
   - Resultados: Adelie vs Chinstrap, Adelie vs Gentoo, Chinstrap vs Gentoo

### 🔍 Evidencia Disponible

#### De Artifacts:

**04a_numeric_summary.json:**
- bill_length: media 43.92 mm, std 5.46
- bill_depth: media 17.15 mm, std 1.97
- flipper_length: media 200.92 mm, std 14.06
- body_mass: media 4201.75 g, std 801.95

**04b_categorical_summary.json:**
- Adelie: n=152 (44.2%)
- Gentoo: n=124 (36.1%)
- Chinstrap: n=68 (19.8%)

**05_visual_registry.json:**
- `box_bill_length_mm_by_species`: Boxplots por especie
- `box_bill_depth_mm_by_species`: Boxplots por especie
- `hist_body_mass_g_by_species`: Histogramas estratificados por especie
- `hist_bill_length_mm_by_species`: Distribuciones completamente separadas

#### Observaciones del Estudiante (Phase 2):

> *"Mira el boxplot box_bill_length_by_species. ¿Observas que los tres rectángulos están en posiciones diferentes?"*
>
> **Respuesta:** "Las distribuciones de longitud del pico son diferentes entre especies. Es decir, hay diferencias morfológicas claras."

### ✅ Decisión y Próximos Pasos

**Confianza:** ALTA  
**Acción:** Testear con ANOVA + Kruskal-Wallis + post-hoc

#### Escenario 1: Si p-value < 0.05 (rechazar H₀)
- Conclusión: Las especies difieren significativamente
- Acción: Proceder a comparaciones pairwise (Tukey)
- Interpretación: Species es un predictor fuerte de tamaño corporal

#### Escenario 2: Si p-value ≥ 0.05 (no rechazar H₀)
- Conclusión: No hay evidencia de diferencias (poco probable dado los gráficos)
- Acción: Revisar supuestos, considerar transformaciones de datos
- Interpretación: Varianza intra-específica es mayor de lo esperado

---

## Hipótesis 2: Simpson's Paradox

### 🎯 Enunciado

> **La correlación global entre flipper_length_mm y body_mass_g (r = 0.871) es artificialmente elevada debido a que Gentoo es más grande tanto en aletas como en masa corporal. Cuando se analiza por especie, la correlación será más débil dentro de cada grupo.**

**ID:** H2  
**Origen:** Detección de paradoja de Simpson en datos agregados  
**Tipo:** Hipótesis de confusión/agregación

### 📊 Variables de Interés

| Variable | Tipo | Rol |
|----------|------|-----|
| `flipper_length_mm` | Numérica | Variable X (predictor) |
| `body_mass_g` | Numérica | Variable Y (respuesta) |
| `species` | Categórica | Variable de estratificación |

### 🧠 Mecanismo Subyacente

**Agregación:** Global, todas las especies mezcladas
- Gentoo: aletas LARGAS (media ~217 mm) + peso ALTO (media ~5076 g)
- Adelie: aletas CORTAS (media ~190 mm) + peso BAJO (media ~3700 g)
- Chinstrap: aletas INTERMEDIAS + peso INTERMEDIO

**Resultado:** Nube de puntos "estirada" diagonalmente → r arti­ficialmente alto

**Estratificación:** Por especie individual
- Dentro de Gentoo: variabilidad menor de tamaño
- Dentro de Adelie: variabilidad menor de tamaño
- Correlación intra-grupal probablemente más débil que global

### 📈 Pruebas Estadísticas Sugeridas

1. **Correlación de Pearson por especie**
   - Calcular r(Adelie), r(Chinstrap), r(Gentoo) por separado
   - Comparar contra r_global = 0.871
   - Si todos < 0.871, confirmar Simpson's Paradox

2. **Regresión lineal: global vs con species**
   - Modelo 1: `body_mass ~ flipper_length` (global)
   - Modelo 2: `body_mass ~ flipper_length + species` (con factor)
   - Modelo 3: `body_mass ~ flipper_length * species` (con interacción)
   - Comparar R², AIC, BIC

3. **ANCOVA: Test de homogeneidad de pendientes**
   - H₀: Las pendientes son iguales entre species
   - H₁: Las pendientes difieren
   - Si p < 0.05, confirmar que la relación varía por especie

4. **Visualización estratificada**
   - Scatterplot global con hue=species + línea de regresión por especie
   - Contrastar con línea de regresión global

### 🔍 Evidencia Disponible

**04d_correlation_pearson__DESCRIBE__20260315_221327.json:**
- flipper_length_mm ↔ body_mass_g: r = **0.871** (correlación MUY fuerte)
- Matriz Pearson completa para inspeccionar patrones

**05_visual_registry.json:**
- `scatter_bill_length_mm_vs_flipper_length_mm_hue_species`: Scatterplot con clusters por especie
- Evidencia visual de agrupamiento → explicación de alta r global

**Observaciones del Estudiante (Phase 3):**

> *"¿Qué hipótesis podrías formular sobre esta relación? ¿Esperas que la correlación sea similar o diferente por especie?"*
>
> **Respuesta:**
> - "La correlación de 0.87 es muy robusta a nivel general"
> - "**No necesariamente será similar por especie** — el efecto de agrupación (Simpson's Paradox)"
> - "Cuando analizamos el conjunto completo, la r se ve reforzada porque Gentoo es grande en todo"
> - "Si aislo una sola especie, podría encontrar r más débil"

### ✅ Decisión y Próximos Pasos

**Confianza:** ALTA (evidencia visual clara de agrupamiento)  
**Acción:** Testear correlaciones por especie + ANCOVA

#### Escenario 1: Si r(Adelie), r(Chinstrap), r(Gentoo) < 0.871
- Conclusión: Simpson's Paradox confirmado
- Interpretación: La correlación global es un artefacto de mezcla de poblaciones
- Recomendación: **SIEMPRE estratificar por species en análisis posteriores**

#### Escenario 2: Si r(especie) ≈ r(global)
- Conclusión: La correlación es robusta incluso dentro de especies
- Interpretación: flipper_length es buen predictor de body_mass en todos los grupos
- Implicación: Relación biológica fundamental, no confusión

---

## Hipótesis 3: Sexo como Confosor

### 🎯 Enunciado

> **El sexo es un confosor importante que afecta tanto la masa corporal como las medidas morfométricas. Un modelo de predicción de body_mass_g que ignore el sexo tendrá sesgo sistemático: sobreestimará hembras, subestimará machos.**

**ID:** H3  
**Origen:** Identificación de variable confusora  
**Tipo:** Hipótesis de confusión (confounder)

### 📊 Variables de Interés

| Variable | Tipo | Rol |
|----------|------|-----|
| `sex` | Categórica (2 niveles + NA) | Variable confusora |
| `body_mass_g` | Numérica | Respuesta principalmente afectada |
| `flipper_length_mm` | Numérica | Respuesta secundaria |
| Otras medidas | Numérica | Respuestas secundarias |

### 🧠 Mecanismo de Confusión

**Dimorfismo sexual:** Machos típicamente > Hembras en masa corporal

**Sesgo sistemático sin sexo:**
1. Modelo predice: ŷ = β₀ + β₁·flipper_length
2. Predicción usa media poblacional (mezcla M/F)
3. Males reales > ŷ promedio → error positivo consistent
4. Females reales < ŷ promedio → error negativo sistemático
5. Residuales correlacionados con sex → violación de supuestos

**Confusión visual:**
- Multimodalidad en body_mass_g: ¿3 picos (1 por especie) o 6 (3 especies × 2 sexos)?
- Si es 6, sexo explica parte de la varianza

### 📈 Pruebas Estadísticas Sugeridas

1. **t-test independiente: Male vs Female**
   - Variable: body_mass_g
   - Gruoups: sex (Male n=168, Female n=165, NA n=11)
   - H₀: μ(Male) = μ(Female)
   - H₁: μ(Male) ≠ μ(Female)
   - Si p < 0.05, sexo causa diferencias sistemáticas

2. **Regresión lineal simple con sexo**
   - Modelo: `body_mass_g ~ sex`
   - R² indica qué porcentaje de varianza explica sexo solo

3. **Regresión múltiple sin sex vs con sex**
   - Modelo 1: `body_mass_g ~ flipper_length_mm`
   - Modelo 2: `body_mass_g ~ flipper_length_mm + sex`
   - Comparar R², AIC, BIC
   - Si R₂ > R₁ significativamente, sex es confosor

4. **Regresión incluida species:**
   - Modelo 3: `body_mass_g ~ flipper_length_mm + species + sex`
   - Evaluar si sex sigue siendo significativo

### 🔍 Evidencia Disponible

**04b_categorical_summary.json:**
- Male: n=168 (48.84%)
- Female: n=165 (47.97%)
- NA (missing): n=11 (3.20%)
- ✅ Sexo está casi perfectamente balanceado

**02_missing_report.json:**
- sex: 11 missing (3.2%)
- species: 0 missing
- island: 0 missing
- Contraste: research pudo identificar species/island pero no sexo en 11 casos
- → Limitación de observación en campo (no MCAR, probablemente MAR)

**Observaciones del Estudiante (Phase 3):**

> *"¿Crees que la especie es suficiente para explicar las diferencias en tamaño, o hay algo más?"*
>
> **Respuesta:**
> - "**La especie es el predictor principal, pero no es suficiente**"
> - "Hay otros factores 'intercalados' que complican la imagen"
> - "Para un modelo de regresión lineal preciso, **ignorar el sexo resultaría en error sistemático (sesgo)**"
> - "Tu modelo fallaría constantemente al predecir el peso de hembras (sobreestimándolas) y machos (subestimándolos)"

### ✅ Decisión y Próximos Pasos

**Confianza:** ALTA (dimorfismo sexual es bien documentado en aves)  
**Acción:** Incluir sex en modelos inferenciales

#### Escenario 1: Si t-test p < 0.05 Y R₂(Modelo 2) >> R₂(Modelo 1)
- Conclusión: Sexo es confosor importante
- Recomendación: **SIEMPRE incluir sex en modelos predictivos**
- Implicación: Los 11 missing requieren imputación o exclusión según objetivo

#### Escenario 2: Si sex no es significativo tras incluir species
- Conclusión: Species podría sumar todo el efecto de sexo (confusión por mezcla)
- Acción: Testear interaction species × sex
- Interpretación: Dimorfismo sexual difiere entre species

---

## Hallazgos Descriptivos

### 📊 Datos Crudos

| Descripción | Valor | Fuente |
|-------------|-------|--------|
| **Total de observaciones** | 344 | 00_raw_profile |
| **Observaciones completas** | 342 (99.4%) | 04a_numeric_summary |
| **Filas duplicadas** | 0 | 03_duplicates_report |

### 📍 Distribución de Muestras

| Especie | n | % | Fuente |
|---------|---|---|--------|
| **Adelie** | 152 | 44.2% | 04b_categorical_summary |
| **Gentoo** | 124 | 36.1% | 04b_categorical_summary |
| **Chinstrap** | 68 | 19.8% | 04b_categorical_summary |

| Isla | n | % |
|-----|---|---|
| **Biscoe** | 168 | 48.8% |
| **Dream** | 124 | 36.1% |
| **Torgersen** | 52 | 15.1% |

### 📌 Datos Faltantes

| Variable | Missing | % | Interpretación |
|----------|---------|---|-----------------|
| bill_length_mm | 2 | 0.58% | Patrón estructural: 2 obs sin medidas |
| bill_depth_mm | 2 | 0.58% | Patrón estructural: 2 obs sin medidas |
| flipper_length_mm | 2 | 0.58% | Patrón estructural: 2 obs sin medidas |
| body_mass_g | 2 | 0.58% | Patrón estructural: 2 obs sin medidas |
| **sex** | **11** | **3.20%** | Limitación de observación en campo |

### 📊 Resumen Estadístico de Variables Numéricas

| Variable | Media | Mediana | Std Dev | Min | Max | Rango |
|----------|-------|---------|---------|-----|-----|-------|
| **bill_length_mm** | 43.92 | 44.45 | 5.46 | 32.1 | 59.6 | 27.5 |
| **bill_depth_mm** | 17.15 | 17.30 | 1.97 | 13.1 | 21.5 | 8.4 |
| **flipper_length_mm** | 200.92 | 197.00 | 14.06 | 172 | 231 | 59 |
| **body_mass_g** | 4201.75 | 4050.00 | 801.95 | 2700 | 6300 | 3600 |

**Interpretación:**
- Distribuciones aproximadamente simétricas (skewness < |0.5|)
- flipper_length y body_mass ligeramente sesgadas a la derecha
- Rangos anchos sugieren considerable variabilidad inter-individual

---

## Patrones Visuales

### 1️⃣ Boxplots por Especie

**Fuentes:** `box_bill_length_mm_by_species`, `box_bill_depth_mm_by_species`, etc.

> *"Observa que los tres rectángulos están en posiciones diferentes"* — Estudiante

**Hallazgo:** Las medianas (línea central de cada caja) están claramente separadas entre especies para todas las variables morfométricas.

**Implicación:** Diferencias biológicas reales entre especies, no artefacto de muestreo.

### 2️⃣ Histogramas Estratificados

**Fuentes:** `hist_flipper_length_mm_by_species`, `hist_body_mass_g_by_species`

**Patrón:**
- Gentoo: concentrado en valores altos (separate cluster)
- Adelie: concentrado en valores bajos
- Chinstrap: rango intermedio con solapamiento parcial

**Multimodalidad observada:**
- body_mass_g es claramente multimodal (3 picos)
- Cada pico corresponde a UNA especie
- Multimodalidad **completamente explicada por species**

> *"¿Ves múltiples picos o es unimodal? Si ves múltiples picos, ¿qué variable explica eso?"*
>
> **Respuesta:** "Claramente multimodal (varios picos). Esta multimodalidad se explica por la variable species. Gentoo se concentra en valores más altos, Adelie y Chinstrap en bajos-intermedios."

### 3️⃣ Scatterplots Bivariados con Hue=Species

**Fuentes:** 
- `scatter_bill_length_mm_vs_flipper_length_mm_hue_species`
- `scatter_bill_depth_mm_vs_flipper_length_mm_hue_species`

**Patrón:** Tres clusters CLARAMENTE separados
- Cada especie ocupa una región distinta del plano
- Solapamiento MINIMAL entre clusters
- Cada cluster sigue una tendencia linear propia (posibles pendientes diferentes)

**Implicación:**
- Species es el predictor más fuerte de morfologia
- Nube global está "estirada" por separación de especies (Simpson's Paradox plausible)

### 4️⃣ Matriz de Correlación (Heatmap Pearson)

**Fuente:** `04d_correlation_pearson__DESCRIBE__20260315_221327.json`

| Par | r | Fuerza | Patrón |
|-----|---|--------|--------|
| **flipper_length ↔ body_mass** | **0.871** | **MUY FUERTE** | Rojo intenso |
| bill_length ↔ flipper_length | 0.656 | Moderada | Rojo |
| flipper_length ↔ bill_depth | -0.584 | Moderada neg | Azul |
| body_mass ↔ bill_depth | -0.472 | Moderada neg | Azul |
| bill_length ↔ bill_depth | -0.234 | Débil neg | Levemente azul |

**Patrones:**
- **Tamaño corporal general:** Medidas grandes correlacionadas positivamente (flipper-masa, bill_length-flipper)
- **Trade-offs:** bill_depth se correlaciona NEGATIVAMENTE con flipper y masa (picos profundos en pingüinos pequeños)

**Hipótesis generada:** Trade-off adaptativo entre evolución del tamaño corporal vs. especialización de pico

---

## Próximos Pasos y Pruebas Estadísticas

### 🔬 Para Testear H1: Diferencias Intra-Específicas

```
ANOVA de una vía: bill_length_mm ~ species
ANOVA de una vía: bill_depth_mm ~ species
ANOVA de una vía: flipper_length_mm ~ species
ANOVA de una vía: body_mass_g ~ species

Si p < 0.05 en todos:
  → Post-hoc Tukey HSD para cada variable
  → Determinar pares Adelie-Chinstrap, Adelie-Gentoo, Chinstrap-Gentoo
```

### 🔬 Para Testear H2: Simpson's Paradox

```
Correlación Pearson en Adelie:  r1 = cor(flipper, mass) [n ≈ 152]
Correlación Pearson en Chinstrap: r2 = cor(flipper, mass) [n ≈ 68]
Correlación Pearson en Gentoo: r3 = cor(flipper, mass) [n ≈ 124]
Correlación Global:           r_global = 0.871

ANCOVA: body_mass ~ flipper_length + species
  → Test de homogeneidad de pendientes
```

### 🔬 Para Testear H3: Sexo como Confosor

```
t-test independiente: body_mass_g ~ sex
  H₀: μ(Male) = μ(Female)

Regresión simple:  body_mass_g ~ flipper_length_mm
  → Captura R₁²

Regresión múltiple: body_mass_g ~ flipper_length_mm + sex
  → Captura R₂²
  
Si R₂ - R₁ > e.g., 0.05 → sex explica varianza adicional importante
```

### 📦 Modelo Integrado Sugerido

```
body_mass_g ~ flipper_length_mm + species + sex
  
  Justificación:
  - species captura diferencias de tamaño entre grupos
  - flipper_length_mm captura variación dentro de grupo
  - sex captura dimorfismo sexual
  
  Predicción: R² > 0.95
```

---

## ✅ Conclusión General

### 🎯 Resumen

El dataset Palmer Penguins exhibe **estructura biológica clara** impulsada predominantemente por **diferencias entre especies**, con contribuciones importantes de **dimorfismo sexual** y posibles **trade-offs morfológicos**.

### 🔑 Puntos Clave

1. **Species es el predictor dominante** de varianza en todas las medidas morfométricas
   - H1: Altamente probable de ser verdadera

2. **Simpson's Paradox probablemente presente** en la correlación global flipper-masa
   - H2: Altamente probable de ser verdadera
   - Recomendación: Estratificar SIEMPRE por species

3. **Sexo es confosor importante** que debe incluirse en modelos inferenciales
   - H3: Probable de ser verdadera
   - Recomendación: Incluir sex en modelos predictivos

4. **Trade-offs adaptativos evidentes:**
   - Gentoo: Grande, aletas largas, picos no particularmente profundos → estrategia "tamaño"
   - Chinstrap: Picos profundos, tamaño medio → estrategia "especialización"
   - Adelie: Pequeña, morfología balanceada → estrategia "agilidad"

### 📋 Recomendaciones para Análisis Futuro

✅ **Regresión lineal múltiple** con species + sex + medidas morfométricas  
✅ **Estratificación por species** en todos los análisis bivariados  
✅ **Imputación o exclusión** de sexo faltante según objetivo  
✅ **PCA** para reducción de dimensionalidad en medidas morfométricas  
✅ **Clasificación** (kNN, SVM, Random Forest) para predicción de species basado en medidas  
✅ **Análisis de nichos morfológicos** para caracterizar diferenciación de especies

### 🗂️ Artifacts Generados

| Artifact | Contenido |
|----------|-----------|
| 06_hypotheses_log | 3 hipótesis falsables con tests sugeridos |
| 07_conclusions | Hallazgos, patrones visuales, próximos pasos |
| Este documento | Síntesis integrada completa |

---

**Generado:** Abril 2, 2026  
**Framework:** Data Observatory  
**Reproducibilidad:** Random seed = 42  
**Fase:** 3 / HYPOTHESIZE_AND_CONCLUDE  

---

## 📚 Apéndice: Referencias a Artifacts

### Artifacts OBSERVE (Phase 1)
- `00_raw_profile__OBSERVE__20260309_125410.json` — Perfil crudo
- `01_schema__OBSERVE__20260309_125410.json` — Esquema de tipos
- `02_missing_report__OBSERVE__20260309_125410.json` — Datos faltantes
- `03_duplicates_report__OBSERVE__20260309_125410.json` — Duplicados

### Artifacts DESCRIBE (Phase 2)
- `04a_numeric_summary__DESCRIBE__20260315_221327.json` — Estadísticas numéricas
- `04b_categorical_summary__DESCRIBE__20260315_221327.json` — Conteos categóricos
- `04d_correlation_pearson__DESCRIBE__20260315_221327.json` — Matriz Pearson
- `05_visual_registry__DESCRIBE__20260315_221328.json` — Registro de gráficos

### Visualizaciones
Carpeta: `artifacts/plots/`
- Boxplots, histogramas, scatterplots, heatmap de correlación
