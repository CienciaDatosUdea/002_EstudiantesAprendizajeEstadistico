# 📊 Resumen de Fases — Lab_01_pinguinos
**Data Observatory | Análisis Exploratorio Reproducible**  
**Universidad de Antioquia — 2026-1**  
**Estudiante: Moran_23**

---

## 📋 Índice de Contenidos

1. [Phase 0: Pre-Análisis](#phase-0-pre-análisis)
2. [Phase 1: OBSERVE](#phase-1-observe)
3. [Phase 2: DESCRIBE](#phase-2-describe)
4. [Phase 3: HYPOTHESIZE and CONCLUDE](#phase-3-hypothesize-and-conclude)

---

## Phase 0: Pre-Análisis

### Objetivo
Forzar el pensamiento analítico ANTES de ejecutar el runner. El estudiante formula expectativas sobre los datos sin haber visto estadísticas.

### Predicciones Iniciales del Estudiante

#### 1. ¿Qué es el dataset?
- Los datos contienen observaciones sobre un fenómeno o población específica
- Cada fila probablemente representa observaciones individuales o mediciones
- Las variables describen características de cada observación

#### 2. ¿Qué patrones esperas encontrar?
- **Esperativa principal:** Correlaciones entre variables
- **Agrupaciones:** Observaciones similares formando clusters
- **Diferencias:** Variación entre distintos grupos dentro de los datos

#### 3. ¿Qué variables crees que están relacionadas?
- Variables que miden propiedades similares del mismo fenómeno
- Cambios en una variable reflejarían cambios en otra
- Trade-offs o compensaciones entre características

#### 4. ¿Qué te sorprendería?
- **Sorpresa:** Ausencia de patrones claros o relaciones
- **Confirmación:** Tendencias consistentes, correlaciones robustas, agrupaciones significativas

### Resultado
✅ **Las predicciones fueron mayormente confirmadas:**
- Se encontraron patrones claros impulsados por especies
- Correlaciones robustas entre medidas morfométricas
- Agrupaciones evidentes en el espacio multivariado

---

## Phase 1: OBSERVE

### Objetivo
Caracterizar la estructura cruda del dataset sin estadísticas descriptivas aún. Entender qué tenemos: filas, columnas, tipos de datos, faltantes.

### Hallazgos Clave

#### Estructura Básica
| Métrica | Valor |
|---------|-------|
| Total de filas | 344 |
| Columnas | 7 |
| Filas con datos completos | 342 |
| Duplicados | 0 |

#### Variables y Tipos

**Categóricas (sin faltantes):**
- `species`: 3 valores (Adelie, Gentoo, Chinstrap)
- `island`: 3 valores (Biscoe, Dream, Torgersen)

**Numéricas:**
- `bill_length_mm` — 2 faltantes
- `bill_depth_mm` — 2 faltantes
- `flipper_length_mm` — 2 faltantes
- `body_mass_g` — 2 faltantes

**Categórica (parcialmente faltante):**
- `sex`: 2 valores (Male, Female) + **11 faltantes (3.2%)**

### Patrones de Datos Faltantes

#### Medidas Morfométricas
- **Patrón:** Exactamente 2 faltantes en CADA variable de medida
- **Interpretación:** No son fallas de instrumento independientes
- **Conclusión:** Dos observaciones completas carecen de todas las medidas físicas

#### Variable `sex`
- **Faltantes:** 11 (3.2% del dataset)
- **Contraste:** `species` e `island` tienen 0 faltantes
- **Interpretación:** Limitación metodológica en campo
- **Hipótesis:** Dificultad de observación remota para determinar sexo sin acercamiento cercano

### Reflexiones del Estudiante

**P1:** *¿Son los 2 faltantes en medidas un patrón estructural?*
- **R:** Sí, sugiere dos observaciones completas sin ninguna medida física (no errores aleatorios)

**P2:** *¿Qué limita el registro de sexo?*
- **R:** Búsqueda remota en campo no permite acercamiento suficiente para sexar

**P3:** *¿Esperas balance 50/50 en sexo?*
- **R:** Predicción: Sí, aproximadamente balanceadas

---

## Phase 2: DESCRIBE

### Objetivo
Computar estadísticas descriptivas y generar visualizaciones exploratorias sin sesgo predefinido.

### Estadísticas Numéricas (n = 342 completos)

| Variable | Media | Mediana | Std Dev | Min | Max | Skewness |
|----------|-------|---------|---------|-----|-----|----------|
| **bill_length_mm** | 43.92 | 44.45 | 5.46 | 32.1 | 59.6 | 0.053 |
| **bill_depth_mm** | 17.15 | 17.3 | 1.97 | 13.1 | 21.5 | -0.143 |
| **flipper_length_mm** | 200.92 | 197.0 | 14.06 | 172 | 231 | 0.346 |
| **body_mass_g** | 4201.75 | 4050 | 801.95 | 2700 | 6300 | 0.470 |

**Interpretación:** Distribuciones aproximadamente simétricas (skewness |< 0.5|), con flipper_length y body_mass ligeramente sesgadas a la derecha.

### Resumen Categórico

#### Especies
| Especie | Conteo | Porcentaje |
|---------|--------|-----------|
| **Adelie** | 152 | 44.2% |
| **Gentoo** | 124 | 36.1% |
| **Chinstrap** | 68 | 19.8% |

#### Sexo (n = 333 con datos)
| Sexo | Conteo | Porcentaje |
|------|--------|-----------|
| **Male** | 168 | 48.8% |
| **Female** | 165 | 48.0% |
| **NA** | 11 | 3.2% |

✅ **Confirmación:** Las categorías de sexo están casi perfectamente balanceadas (coincide con la predicción del estudiante)

### Matriz de Correlación (Pearson)

| Par de Variables | Correlación | Fuerza | Interpretación |
|------------------|-------------|--------|----------------|
| **flipper_length ↔ body_mass** | **0.871** | Muy fuerte | Pingüinos con aletas largas pesan significativamente más |
| **bill_length ↔ flipper_length** | 0.656 | Moderada | Picos largos asociados con aletas largas |
| **bill_depth ↔ flipper_length** | -0.584 | Moderada negativa | Trade-off: picos profundos con aletas más cortas |
| **bill_depth ↔ body_mass** | -0.472 | Moderada negativa | Pingüinos más pesados tienen picos menos profundos |

**Patrón:** Estructura de tamaño corporal general (medidas grandes correlacionadas positivamente) con trade-offs morfológicos (especialmente en profundidad de pico).

### Visualizaciones Clave

#### 1. **Distribuciones de Conteo**
- Species: Adelie domina (44%)
- Island: Biscoe tiene mayoría
- Sex: Balanceado 50/50

#### 2. **Boxplots por Especie**
- **bill_length_mm:** Tres distribuciones claramente separadas
- **bill_depth_mm:** Chinstrap y Adelie más profundos que Gentoo
- **Conclusión:** Diferencias morfológicas significativas entre especies

#### 3. **Histogramas Univariados**
- **flipper_length:** Gentoo separada (aletas más largas)
- **body_mass:** MULTIMODAL con 3 picos claros (uno por especie)
- **Conclusión:** Cada especie ocupa un rango distinto de tamaño

#### 4. **Scatterplots**
- **bill_length vs flipper_length:** 3 clusters sin solapamiento
- **bill_depth vs flipper_length:** Trade-off negativo visible (Adelie/Chinstrap vs Gentoo)
- **Conclusión:** Especies en nichos morfológicos distintos

#### 5. **Heatmap de Correlación**
- Rojo = correlaciones positivas fuertes
- Azul = correlaciones negativas
- El par flipper–masa (0.871) es el más rojo (correlación más fuerte)

### Reflexiones del Estudiante

**P1:** *¿Qué sugiere que los boxplots de species esté en posiciones diferentes?*
- **R:** Las distribuciones de longitud del pico son diferentes entre especies. Diferencias morfológicas claras.

**P2:** *¿Cuál par tiene la correlación más fuerte en el heatmap?*
- **R:** flipper_length_mm ↔ body_mass_g. Relación lineal positiva muy fuerte. Cuando aumenta la aleta, aumenta el peso.

**P3:** *¿La multimodalidad en body_mass es unimodal o multimodal?*
- **R:** Claramente multimodal (3 picos). Explicada por species: Gentoo concentrada en valores altos; Adelie y Chinstrap en valores bajos/intermedios.

---

## Phase 3: HYPOTHESIZE and CONCLUDE

### Objetivo
Generar **hipótesis falsables** y **conclusiones estratificadas** fundamentadas en evidencia de los artifacts.

### Hipótesis 1: Diferencias Intra-Específicas

**Enunciado:**
> Las tres especies de pingüino difieren significativamente en sus medidas morfométricas. Gentoo es la más grande en general; Chinstrap tiene picos relativamente más profundos; Adelie es la más pequeña.

**Variables de interés:** `species` → `bill_length`, `bill_depth`, `flipper_length`, `body_mass`

**Perfiles por Especie:**

| Especie | bill_length | bill_depth | flipper_length | body_mass |
|---------|-------------|----------|-----------------|-----------|
| **Adelie** | Corta (~38.8 mm) | Moderada-Alta (~18.7 mm) | Corta (~190 mm) | Baja (~3700 g) |
| **Chinstrap** | **MÁS LARGA** (~49.6 mm) | **MÁS PROFUNDA** (~18.4 mm) | Intermedia (~195 mm) | Media (~3733 g) |
| **Gentoo** | Moderada (~47.3 mm) | Baja (~15 mm) | **MÁS LARGA** (~217 mm) | **MÁS ALTA** (~5076 g) |

**Trade-offs Adaptativos:**
- **Adelie:** Pequeña y ágil
- **Chinstrap:** Especializada en picos profundos (estrategia de alimentación distinta)
- **Gentoo:** Invierte en aletas largas y masa corporal (estrategia de tamaño)

**Tests Sugeridos:** ANOVA, Kruskal-Wallis, Tukey HSD post-hoc

**Evidencia:**
- Boxplots muestran separación clara de medianas (box_bill_length_by_species, etc.)
- Histogramas revelan distribuciones completamente separadas (especialmente body_mass_g)
- Scatterplots = 3 clusters distintos sin solapamiento significativo

---

### Hipótesis 2: Simpson's Paradox en la Correlación

**Enunciado:**
> La correlación global entre flipper_length y body_mass (r = 0.871) es artificialmente elevada porque Gentoo es más grande TANTO en aletas COMO en masa. Cuando se analiza por especie, la correlación será más débil dentro de cada grupo.

**Razonamiento del Estudiante:**
- Correlación alta se ve reforzada porque Gentoo es "grande en todo"
- Los Adelie son "pequeños en todo"
- Esto "estira" la nube de puntos globalmente
- **Efecto:** Correlación artificialmente alta en conjunto completo

**Variabilidad Intra-Específica:**
- Dentro de una misma especie, variabilidad de tamaño es menor
- Otros factores (nutrición, edad, sexo) influyen en peso sin cambiar longitud ósea
- Especies homogéneas → correlación más débil que global
- Dimorfismo sexual en algunas especies → correlación se mantiene alta solo si tamaño es diferenciador principal

**Tests Sugeridos:** Correlaciones por especie, ANCOVA, test de homogeneidad de pendientes

**Evidencia:**
- Correlación global Pearson = 0.871 (muy fuerte)
- Scatterplot (flipper vs body_mass, hue=species) muestra Gentoo como cluster elevado
- Nube global "estirada" por separación de especies

---

### Hipótesis 3: Sexo como Confosor

**Enunciado:**
> El sexo es un confosor importante que afecta tanto masa corporal como medidas morfométricas. Un modelo que ignore sexo tendrá sesgo sistemático: sobreestimará hembras, subestimará machos.

**Mecanismo:**
- Dimorfismo sexual: machos más grandes que hembras en casi todas las especies
- Predictor sin sexo acumula error: predice peso promedio (desviado)
- Males reales > predicción → residual negativo sistémático
- Females reales < predicción → residual positivo sistémático

**Tests Sugeridos:** 
- t-test independiente (sex vs body_mass)
- Regresión múltiple con/sin sexo
- Comparación R² y AIC

**Evidencia:**
- Sex está balanceado (48.8% M, 48% F), pero 3.2% faltante
- Multimodalidad en body_mass podría deberse PARCIALMENTE a sexo (no solo especies)
- Researchers siempre identificaron species/island pero NO sexo en 11 casos → limitación de observación

---

### Análisis Simplificado del Rol de Species vs. Sexo

**Pregunta del Estudiante:**
> ¿La especie EXPLICA suficientemente las diferencias de tamaño? ¿O hay algo más intercalado (sexo, isla)?

**Respuesta:**

**La especie explica un parte GRANDE pero NO SUFICIENTE:**

1. **Especie:** Factor PRINCIPAL
   - Predictor más fuerte de tamaño
   - Clusters visibles en scatterplots
   - Separación de medianas en boxplots

2. **Sexo:** Factor intercalado (confusor)
   - Dimorfismo sexual: machos > hembras
   - Ignorarlo → error sistemático en predicción
   - Modelo preciso de regresión DEBE incluirlo

3. **Isla:** Factor SECUNDARIO
   - Asociada con species (chi² significativo)
   - Probablemente confundida con species
   - Menos importante que especies o sexo

### Conclusión Integrada

**Hallazgos Descriptivos Clave:**
- 344 pingüinos de 3 especies en 3 islas
- Medidas NO independientes: flipper–masa r = 0.871
- Multimodalidad en body_mass COMPLETAMENTE explicada por species
- Tres especies ocupan regiones distintas del espacio morfológico sin solapamiento
- Sexo balanceado (~50/50) pero con 11 faltantes

**Patrones Visuales:**
- Adelie: pequeña, ágil
- Chinstrap: especializada en pico profundo
- Gentoo: grande, aletas largas, bajo pico profundo

**Trade-offs Morfológicos:**
- No todos los pingüinos son "grandes en todo"
- Cada especie invierte en diferentes características
- Estructura clara de nichos ecológicos/morfológicos

**Recomendaciones para Análisis Futuro:**
✅ Usar species como variable categórica PRINCIPAL  
✅ Incluir sexo como confosor en modelos  
✅ Estratificar análisis por species cuando sea apropiado  
✅ ANCOVA para testing de homogeneidad de pendientes  
✅ Considerar imputación de los 11 sexos faltantes o exclusión según objetivo  

---

## 📈 Resumen Ejecutivo

| Fase | Objetivo | Hallazgo Clave |
|------|----------|----------------|
| **0** | Predicción previa | Patrones claros esperados ✓ |
| **1** | Estructura cruda | 344 obs, 7 vars, 2 obs sin medidas, 11 sexos faltantes |
| **2** | Estadísticas y viz | 3 clusters por especie, r=0.871 flipper–masa, trade-offs morfológicos |
| **3** | Hipótesis | H1: species explica diferencias; H2: Simpson's Paradox; H3: sexo es confosor |

### Dataset Listo Para:
- ✅ Regresión lineal múltiple (species + sexo + medidas)
- ✅ Clasificación de especies (kNN, árboles, SVM)
- ✅ Análisis de componentes principales (PCA)
- ✅ Clustering y análisis de nichos morfológicos

---

**Generado:** Abril 2, 2026  
**Framework:** Data Observatory  
**Reproducibilidad:** Random seed = 42
