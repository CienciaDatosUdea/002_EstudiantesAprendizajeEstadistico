# Hipotesis y Conclusiones Documentadas

## Hipotesis

### H1
- Enunciado: `flipper_length_mm is associated with body_mass_g`
- Variables: `flipper_length_mm`, `body_mass_g`
- Pruebas sugeridas: `Pearson`, `Spearman`
- Evidencia:
  - `pearson_corr.json`
  - `spearman_corr.json`
  - `plots/heatmap_pearson_corr.png`
  - `04_descriptive_stats.json`

### H2
- Enunciado: `bill_length_mm differs across species`
- Variables: `bill_length_mm`, `species`
- Pruebas sugeridas: `ANOVA`, `Kruskal`
- Evidencia:
  - `04_descriptive_stats.json`
  - `plots/boxplot_bill_length_mm_by_species.png`
  - `05_visual_registry.json`

### H3
- Enunciado: `species is associated with island`
- Variables: `species`, `island`
- Pruebas sugeridas: `Chi-square`
- Evidencia:
  - `cross_table_species_island.json`
  - `04_descriptive_stats.json`

Fuente: `06_hypotheses_log.json`

## Resultados de pruebas

### Asociacion entre `flipper_length_mm` y `body_mass_g`
- Pearson: estadistico `0.8712017673060113`, p-valor `4.3706809630004724e-107`, `n = 342`
- Spearman: estadistico `0.8399741230312999`, p-valor `2.763218997179657e-92`, `n = 342`

### Diferencia de `bill_length_mm` por `species`
- ANOVA: estadistico `410.6002550405077`, p-valor `2.6946137388895495e-91`, `n_groups = 3`
- Kruskal: estadistico `244.13671803364164`, p-valor `9.691371997194331e-54`, `n_groups = 3`

### Asociacion entre `species` e `island`
- Chi-square: estadistico `299.55032743148195`, p-valor `1.3545738297192517e-63`, `dof = 4`

Fuente: `08_tests.json`

## Conclusiones registradas

### A. Hallazgos descriptivos
- Hay hallazgos descriptivos disponibles para variables numericas, resumenes categoricos y la tabla cruzada `species-island`.
- Evidencia: `07_conclusions.json`, `04_descriptive_stats.json`

### B. Patrones visuales
- El registro visual incluye count plots, histogramas, boxplots y un heatmap de correlacion de Pearson.
- Evidencia: `07_conclusions.json`, `05_visual_registry.json`

### C. Proximas hipotesis a probar
- Evaluar la asociacion entre `flipper_length_mm` y `body_mass_g` con Pearson y Spearman.
- Evaluar si `bill_length_mm` difiere entre especies con ANOVA y Kruskal.
- Evaluar la asociacion entre `species` e `island` con Chi-square.
- Evidencia: `07_conclusions.json`

## Preguntas para el investigador humano

1. Como deben tratarse las filas con valores faltantes de `sex` en analisis estratificados.
2. Si debe aplicarse complete-case handling u otra regla de preprocesamiento antes de pruebas con variables numericas.

Fuente: `09_questions.json`
