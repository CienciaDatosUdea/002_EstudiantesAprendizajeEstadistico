# Reporte final del laboratorio

## 1. Objetivo

Analizar la estructura, la distribución y las relaciones entre variables del dataset de pingüinos, con base en los artifacts generados por el runner y la evidencia estadística asociada.

## 2. Evidencia disponible

Los resultados se apoyan en los siguientes artifacts:

- `00_raw_profile.json`
- `04_descriptive_stats.json`
- `05_visual_registry.json`
- `08_tests.json`

## 3. Perfil del dataset

El perfil crudo indica que el conjunto tiene:
- 344 filas
- 7 columnas
- 0 duplicados
- 11 faltantes en la variable `sex`
- 2 faltantes en cada una de estas variables: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`

Variables detectadas:
- categóricas: `species`, `island`, `sex`
- numéricas: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`

## 4. Resumen descriptivo

### Variables numéricas

| Variable | Media | Desv. Std. | Mín. | Máx. |
|---|---:|---:|---:|---:|
| bill_length_mm | 43.922 | 5.460 | 32.1 | 59.6 |
| bill_depth_mm | 17.151 | 1.975 | 13.1 | 21.5 |
| flipper_length_mm | 200.915 | 14.062 | 172.0 | 231.0 |
| body_mass_g | 4201.754 | 801.955 | 2700.0 | 6300.0 |

### Distribución categórica

- species:
  - Adelie: 152 (44.19%)
  - Gentoo: 124 (36.05%)
  - Chinstrap: 68 (19.77%)

- island:
  - Biscoe: 168 (48.84%)
  - Dream: 124 (36.05%)
  - Torgersen: 52 (15.12%)

- sex:
  - MALE: 168 (48.84%)
  - FEMALE: 165 (47.97%)
  - missing: 11 (3.20%)

## 5. Correlaciones principales

La matriz de correlación Pearson mostró las siguientes asociaciones más relevantes:

- `flipper_length_mm` vs `body_mass_g`: 0.871
- `bill_length_mm` vs `flipper_length_mm`: 0.656
- `bill_length_mm` vs `body_mass_g`: 0.595
- `bill_depth_mm` vs `flipper_length_mm`: -0.584
- `bill_depth_mm` vs `body_mass_g`: -0.472

La relación entre `bill_length_mm` y `flipper_length_mm` fue claramente positiva y estadísticamente relevante.

## 6. Patrones visuales

Los gráficos generados muestran:

- diferencias marcadas de masa corporal por especie,
- composición desigual de especies según isla,
- asociación visual entre longitud del pico y longitud de la aleta,
- correlación positiva general entre las variables morfológicas.

### Archivos visuales generados

- `artifacts/05_species_bar.png`
- `artifacts/05_island_bar.png`
- `artifacts/05_sex_bar.png`
- `artifacts/05_bill_length_mm_hist.png`
- `artifacts/05_bill_depth_mm_hist.png`
- `artifacts/05_flipper_length_mm_hist.png`
- `artifacts/05_body_mass_g_hist.png`
- `artifacts/05_body_mass_g_by_species_boxplot.png`
- `artifacts/05_bill_length_mm_by_island_boxplot.png`
- `artifacts/05_bill_length_mm_vs_flipper_length_mm_by_species_scatter.png`
- `artifacts/05_numeric_correlation_heatmap.png`

## 7. Hipótesis y pruebas

### Hipótesis 1
**Hipótesis:** la masa corporal difiere entre especies.

**Prueba estadística:** ANOVA de una vía.

**Resultado:**
- F = 343.626
- p = 2.8923681333773435e-82

**Interpretación:** la evidencia estadística apoya una diferencia significativa de `body_mass_g` entre especies. No se puede afirmar causalidad, solo asociación estadística.

### Hipótesis 2
**Hipótesis:** la distribución de especies está asociada con la isla.

**Prueba estadística:** chi-cuadrado de independencia.

**Resultado:**
- chi2 = 299.55
- dof = 4
- p = 1.354573829719252e-63

**Interpretación:** la evidencia apoya una asociación entre `species` e `island`. La estructura cruzada muestra que Gentoo aparece casi exclusivamente en Biscoe y Chinstrap casi exclusivamente en Dream.

### Hipótesis 3
**Hipótesis:** existe asociación lineal entre longitud del pico y longitud de la aleta.

**Prueba estadística:** correlación de Pearson.

**Resultado:**
- r = 0.656
- p = 1.7439736176205688e-43

**Interpretación:** hay evidencia estadística de una relación lineal positiva entre `bill_length_mm` y `flipper_length_mm`. No implica causalidad; solo asociación.

## 8. Conclusión general

Los datos muestran una estructura clara y consistente de diferencias morfológicas por especie y una asociación notable entre especie e isla. Además, se observó una correlación positiva entre longitud del pico y longitud de la aleta. Todos los resultados estadísticos son compatibles con la evidencia descriptiva y visual del dataset, pero no permiten afirmar causalidad: solo permiten sostener que existen asociaciones estadísticamente relevantes entre las variables estudiadas.

## 9. Preguntas para un investigador humano

1. ¿La asociación entre especie e isla refleja una distribución ecológica real o un sesgo de muestreo?
2. ¿La diferencia en masa corporal podría responder a factores no observados como edad, dieta o condición fisiológica?
3. ¿La correlación entre longitud del pico y longitud de la aleta es homogénea en todas las especies o varía entre grupos?
4. ¿Debe considerarse `sex` como factor de confusión en comparaciones morfológicas?
5. ¿Conviene controlar la isla al evaluar diferencias entre especies para separar efectos geográficos de diferencia biológica?

## 10. Comparación entre enfoque clásico (notebook manual) y enfoque con agentes

El enfoque clásico en un notebook manual suele ser más lineal y explícito: el analista escribe el código paso a paso, decide qué explorar, ejecuta celdas en orden y valida cada resultado manualmente antes de avanzar. Eso da mucha transparencia sobre qué se hizo en cada bloque, pero también exige más disciplina y vigilancia humana: si se cambia un filtro, una limpieza o una hipótesis, es fácil perder trazabilidad y propagar errores sin darse cuenta. En términos de reproducibilidad, un notebook manual puede ser muy reproducible cuando está bien documentado y se usa un entorno fijo, pero depende mucho de la calidad del proceso manual y de la disciplina del usuario para mantener un registro claro de las decisiones.

El enfoque con agentes, como el que usamos en este laboratorio, cambia el flujo porque la lógica de análisis se articula como una conversación en fases: OBSERVE, DESCRIBE y HYPOTHESIZE_AND_CONCLUDE. El agente propone qué hacer, el runner ejecuta la operación y se generan artifacts (JSON y PNG) que sirven como evidencia formal. Esto aporta ventajas importantes: aumenta la estructura del análisis, reduce la carga cognitiva del analista y mejora la trazabilidad porque cada etapa queda reflejada en archivos específicos. También ayuda a mantener un criterio más homogéneo y a documentar mejor las decisiones. En términos de control de errores, el agente puede ser útil para evitar sesgos de análisis y para reforzar la regla de que las conclusiones deben basarse en evidencia disponible, pero también tiene una limitación importante: depende de la calidad del prompt, del alcance de las herramientas y de la supervisión humana para evitar interpretaciones demasiado amplias o ejecución no deseada.

En resumen, el enfoque clásico ofrece más control directo y más libertad analítica, pero requiere mayor esfuerzo manual y más riesgo de errores de procedimiento. El enfoque con agentes ofrece mayor organización, trazabilidad y automatización del flujo de análisis, pero exige una supervisión explícita para garantizar reproducibilidad, validación y que la evidencia siga siendo rigurosamente la que aparece en los artifacts. En otras palabras, el notebook manual favorece la ejecución detallada y personalizada; el enfoque con agentes favorece la estructura y la documentación, aunque con menos control operativo fino por parte del analista.

## 11. Archivos entregados

- `reporte_final.md`
- `reporte_final.html`
