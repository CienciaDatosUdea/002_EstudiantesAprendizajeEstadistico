# Resumen del Observatorio de Datos: Penguins

## Observacion inicial

El dataset analizado contiene 344 filas y 7 columnas. Las variables numericas son `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` y `body_mass_g`. Las variables categoricas son `species`, `island` y `sex`.

Se identificaron valores faltantes en `bill_length_mm` (2), `bill_depth_mm` (2), `flipper_length_mm` (2), `body_mass_g` (2) y `sex` (11). No se encontraron filas duplicadas. Las variables de baja cardinalidad son `species`, `island` y `sex`.

Fuente: `00_raw_profile.json`

## Descripcion

Los descriptivos numericos y categoricos fueron consolidados en `04_descriptive_stats.json`.

Entre las variables numericas, la asociacion mas fuerte registrada aparece entre `flipper_length_mm` y `body_mass_g`, con correlacion de Pearson `0.8712` y correlacion de Spearman `0.8400`.

Las tablas cruzadas documentadas incluyen la relacion entre `species` e `island`.

Fuente: `04_descriptive_stats.json`

## Visualizaciones

El registro visual incluye count plots para variables categoricas, histogramas para variables numericas, boxplots por especie y un heatmap de correlacion de Pearson.

Visuales destacados:
- `plots/categorical_count_plot_species.png`
- `plots/boxplot_bill_length_mm_by_species.png`
- `plots/heatmap_pearson_corr.png`

Fuente: `05_visual_registry.json`

## Hipotesis documentadas

1. `flipper_length_mm` is associated with `body_mass_g`.
2. `bill_length_mm` differs across `species`.
3. `species` is associated with `island`.

Fuente: `06_hypotheses_log.json`

## Pruebas estadisticas registradas

- Pearson para `flipper_length_mm` y `body_mass_g`
- Spearman para `flipper_length_mm` y `body_mass_g`
- ANOVA para `bill_length_mm` por `species`
- Kruskal para `bill_length_mm` por `species`
- Chi-square para `species` e `island`

Los resultados quedaron documentados en `08_tests.json`.

## Conclusiones y preguntas abiertas

Las conclusiones registradas organizan los hallazgos en tres capas: hallazgos descriptivos, patrones visuales y siguientes hipotesis a probar.

Las preguntas abiertas para un investigador humano se centran en el manejo de valores faltantes de `sex` y en la regla de preprocesamiento para filas con faltantes numericos antes de futuras pruebas.

Fuentes:
- `07_conclusions.json`
- `09_questions.json`
