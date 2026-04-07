## Hipótesis y conclusiones — Observatorio de Datos: Lab 01 Pinguinos

**Fecha:** 2026-04-02  
**Autor:** Palacios_10

---

## Sección de Hipótesis

### H1 — Masa corporal por especie
- **Enunciado:** La mediana de `body_mass_g` es significativamente mayor en `Gentoo` que en `Adelie` y `Chinstrap`.
- **Metodología propuesta:** ANOVA de una vía (si se cumplen supuestos de normalidad y homogeneidad de varianzas) o Kruskal–Wallis (si no), seguido de pruebas post‑hoc (Tukey HSD o Dunn con corrección). Reportar tamaño del efecto (eta² / d de Cohen) y checks de supuestos (Shapiro, Levene).
- **Evidencia:**
  - Estadística descriptiva: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_numeric_summary_20260402_134637.json` (n_total = 344; `body_mass_g` mean ≈ 4201.75; median = 4050; sd ≈ 801.95).
  - Visual: `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_box_body_mass_g_by_species_20260402_134701.png`.
  - Archivo de prueba confirmatoria (pendiente de ejecución): `FASE_3_h1_tests.json` (se espera guardar en `FASE_2_AGENTES/artifacts/` con timestamp: `FASE_3_h1_tests_YYYYMMDD_HHMMSS.json`).
- **Estado:** APOYADA (basada en evidencia descriptiva y visual). Confirmación estadística formal pendiente (ANOVA/Kruskal en FASE_3).

### H2 — Correlación aleta / masa
- **Enunciado:** Existe una asociación positiva entre `flipper_length_mm` y `body_mass_g`.
- **Metodología propuesta:** Coeficiente de correlación de Pearson (si se cumplen supuestos); Spearman si no. Calcular p‑valor, intervalo de confianza y evaluar correlaciones estratificadas por `species`.
- **Evidencia:**
  - Matriz de correlación (Pearson): `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json` (coeficiente global `flipper_length_mm` ↔ `body_mass_g` ≈ 0.8712).
  - Visual de soporte: `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_heatmap_corr_pearson_20260402_134714.png`.
  - Archivo de confirmación por especie (recomendado): `FASE_3_h2_corr_by_species.json`.
- **Resumen numérico:** r ≈ 0.8712 (Pearson, global). p‑value confirmatorio por especie pendiente.
- **Estado:** APOYADA (fuerte correlación global observada). Recomendado: calcular p‑value e IC, y analizar correlación dentro de cada especie (archivo FASE_3 pendiente).

### H3 — Morfología de picos por especie
- **Enunciado:** Las especies difieren en la distribución conjunta de `bill_length_mm` y `bill_depth_mm`.
- **Metodología propuesta:** MANOVA (o pruebas multivariadas no paramétricas) para la comparación conjunta; complementariamente ANOVA/Kruskal univariados para cada dimensión con corrección múltiple; análisis discriminante para cuantificar separabilidad.
- **Evidencia:**
  - Visual: `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_scatter_bill_length_mm_vs_bill_depth_mm_hue_species_20260402_134711.png` (clusters diferenciables por especie).
  - Estadística descriptiva por variable: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_numeric_summary_20260402_134637.json`.
  - Archivo de confirmación estadística (pendiente): `FASE_3_h3_manova.json`.
- **Estado:** APOYADA (evidencia visual y descriptiva consistente). Confirmación multivariante pendiente.

### H4 — Missingness en `sex`
- **Enunciado:** La ausencia de la variable `sex` es independiente de `species`.
- **Metodología propuesta:** Construir tabla de contingencia `species × sex_missing` y aplicar Chi‑cuadrado de independencia; si existen celdas con frecuencia esperada < 5, emplear Fisher exact test. Complementar con test de Little para MCAR y modelado logístico de missingness para explorar MAR/MNAR.
- **Evidencia:**
  - Reporte de faltantes: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json` (11 faltantes en `sex`, ≈ 3.20%).
  - Crosstabs auxiliares: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_crosstab_20260402_134638.json`.
  - Archivo de confirmación estadística (pendiente): `FASE_3_missing_sex_test.json`.
- **Estado:** INCONCLUSA. La evidencia actual es insuficiente para afirmar independencia; requiere el test Chi‑cuadrado/Fisher y análisis de patrones de missingness.

---

## Sección de Conclusiones (Las 3 capas)

### Hallazgos descriptivos
- Volumen de datos: n = 344 observaciones; 7 variables medidas; sin duplicados detectados en el dataset analizado.
- Calidad: Faltantes en `sex` = 11 registros (~3.2%).  
- Resumen de `body_mass_g`: mean ≈ 4201.75; median = 4050; sd ≈ 801.95. (Ver `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_numeric_summary_20260402_134637.json`.)

### Patrones visuales
- Gentoo muestra medianas de masa corporal superiores en comparación con Adelie y Chinstrap (boxplots).  
- El scatter de `bill_length_mm` vs `bill_depth_mm` evidencia agrupamientos por especie, compatibles con separabilidad morfológica.  
- Correlación fuerte entre `flipper_length_mm` y `body_mass_g` (r ≈ 0.8712), que sugiere relación morfológica consistente a nivel global.

### Próximas hipótesis y trabajos sugeridos
- Ejecutar FASE_3 para obtener pruebas confirmatorias (ANOVA/Kruskal, MANOVA, chi‑cuadrado) y registrar artefacts:
  - `FASE_2_AGENTES/artifacts/FASE_3_h1_tests_YYYYMMDD_HHMMSS.json`
  - `FASE_2_AGENTES/artifacts/FASE_3_h2_corr_by_species_YYYYMMDD_HHMMSS.json`
  - `FASE_2_AGENTES/artifacts/FASE_3_h3_manova_YYYYMMDD_HHMMSS.json`
  - `FASE_2_AGENTES/artifacts/FASE_3_missing_sex_test_YYYYMMDD_HHMMSS.json`
- Modelos predictivos para imputación de `sex` con validación cruzada.
- Incluir variables contextuales (años, condiciones climáticas) para análisis multivariado.

---

## Cierre técnico — reproducibilidad y control de alucinación
- El pipeline Agente (planificador) + Runner (ejecutor) proporciona trazabilidad y reproducibilidad: cada inferencia debe apuntar a un artifact JSON/PNG que contenga el test o gráfico subyacente.  
- Política operacional aplicada: no emitir veredictos inferenciales definitivos sin artifact FASE_3 que documente el test (estadístico, p‑valor, tamaño del efecto, supuestos). Esto reduce el riesgo de afirmaciones no sustentadas y facilita auditoría por pares.
- Recomendación: los artefacts FASE_3 deben incluir campos estandarizados: `test_name`, `statistic`, `p_value`, `effect_size`, `assumptions_checked`, `notes`.

---

## Referencias rápidas (archivos usados como evidencia)
- `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_profile_dataset_20260402_134637.json`  
- `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_numeric_summary_20260402_134637.json`  
- `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json`  
- `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json`  
- `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_crosstab_20260402_134638.json`  
- `FASE_2_AGENTES/artifacts/06_hypotheses_log.json`  
- `FASE_2_AGENTES/artifacts/07_conclusions.json`  

---
