# Reporte Final - Lab_01 (Penguins)

## Resumen del dataset
El dataset analizado contiene 344 filas y 7 columnas, con variables categoricas (species, island, sex) y numericas (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) [00_raw_profile__OBSERVE__20260320_151726.json:n_rows] [00_raw_profile__OBSERVE__20260320_151726.json:n_cols] [00_raw_profile__OBSERVE__20260320_151726.json:columns].
El porcentaje total de valores faltantes es 0.79%, concentrado especialmente en sex (11 faltantes, 3.2%), y no se detectaron duplicados [02_missing_report__OBSERVE__20260320_151726.json:pct_missing_overall] [02_missing_report__OBSERVE__20260320_151726.json:by_column.sex.n_missing] [02_missing_report__OBSERVE__20260320_151726.json:by_column.sex.pct_missing] [03_duplicates_report__OBSERVE__20260320_151726.json:n_duplicates].

## Hallazgos descriptivos
- En variables numericas, la asociacion lineal mas alta reportada es entre flipper_length_mm y body_mass_g (r = 0.8712) [04d_correlation_pearson__HYPOTHESIZE__20260320_153631.json:matrix.flipper_length_mm.body_mass_g].
- En variables categoricas, species tiene tres categorias con conteos 152 (Adelie), 124 (Gentoo) y 68 (Chinstrap) [04b_categorical_summary__DESCRIBE__20260320_151829.json:species.value_counts].
- El cruce species-island muestra concentracion por pares especificos: Gentoo en Biscoe (124), Chinstrap en Dream (68) y Adelie en Torgersen (52) [04c_crosstab_species_island__HYPOTHESIZE__20260320_153631.json:table.Biscoe.Gentoo] [04c_crosstab_species_island__HYPOTHESIZE__20260320_153631.json:table.Dream.Chinstrap] [04c_crosstab_species_island__HYPOTHESIZE__20260320_153631.json:table.Torgersen.Adelie].

## Patrones visuales
- Los scatter de bill_length_mm vs flipper_length_mm y bill_depth_mm vs flipper_length_mm muestran separacion visual por especie [05_visual_registry__DESCRIBE__20260320_151832.json:plots[8].plot_id] [05_visual_registry__DESCRIBE__20260320_151832.json:plots[9].plot_id].
- El scatter de flipper_length_mm vs body_mass_g por especie muestra agrupacion por especie y tendencia positiva global [05_visual_registry_hypothesize__HYPOTHESIZE__20260320_153631.json:plots[0].plot_id] [04d_correlation_pearson__HYPOTHESIZE__20260320_153631.json:matrix.flipper_length_mm.body_mass_g].
- El boxplot de bill_length_mm por especie sugiere diferencias entre grupos [05_visual_registry__DESCRIBE__20260320_151832.json:plots[6].plot_id].

## Comparacion con pre-analisis del estudiante
En Phase 0, el estudiante anticipaba asociaciones entre rasgos morfometricos, interes en posibles diferencias por especie y curiosidad sobre el rol de island y sex [student_log.md:Phase 0 - Pre-Analysis].
En Phase 2, el estudiante observo sectorizacion por especie en scatter y destaco la relacion masa-aleta, proponiendo pruebas basicas [student_log.md:Phase 2 - Describe Reflection].
Los resultados actuales son consistentes con esas expectativas iniciales: hay evidencia descriptiva de separacion por especie en graficos y alta asociacion global entre flipper_length_mm y body_mass_g [05_visual_registry__DESCRIBE__20260320_151832.json:plots[8].plot_id] [05_visual_registry_hypothesize__HYPOTHESIZE__20260320_153631.json:plots[0].plot_id] [04d_correlation_pearson__HYPOTHESIZE__20260320_153631.json:matrix.flipper_length_mm.body_mass_g].

## Hipotesis propuestas
- H1: flipper_length_mm esta asociado positivamente con body_mass_g.
- H2: bill_length_mm difiere entre especies.
- H3: species esta asociada con island.
(Ver detalle formal en 06_hypotheses_log__HYPOTHESIZE_AND_CONCLUDE__20260320_154100.json.)

## Que probar despues
1. Correlaciones por especie para flipper_length_mm vs body_mass_g (en lugar de solo correlacion global).
2. Prueba chi-cuadrado para species vs island sobre la tabla de contingencia generada.
3. Prueba ANOVA o Kruskal-Wallis para bill_length_mm por species.

## Resultados de pruebas estadisticas
- H1 (flipper_length_mm vs body_mass_g):
	- Pearson: estadistico = 0.8712017673060113, p-valor = 4.3706809630006207e-107, decision = rechazar_H0.
	- Spearman: estadistico = 0.8399741230312999, p-valor = 2.763218997179663e-92, decision = rechazar_H0.
- H2 (bill_length_mm entre especies):
	- ANOVA: estadistico = 410.6002550405077, p-valor = 2.6946137388895146e-91, decision = rechazar_H0.
	- Kruskal-Wallis: estadistico = 244.13671803364164, p-valor = 9.691371997194331e-54, decision = rechazar_H0.
- H3 (species vs island):
	- Chi-cuadrado: estadistico = 299.55032743148195, p-valor = 1.354573829719252e-63, gl = 4, decision = rechazar_H0.
	- Tamaño de efecto: Cramer's V = 0.6598431008795325.

Fuente: [08_tests__HYPOTHESIZE_AND_CONCLUDE__20260320_154944.json].
