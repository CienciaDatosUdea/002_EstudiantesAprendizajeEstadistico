# Observatorio de Datos — Pingüinos

Generado: 2026-05-02T18:35:40.153867


## Observación inicial

- Filas: 344
- Columnas: 7
- Variables: species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex

Valores faltantes:
  - bill_length_mm: 2 (0.58%)
  - bill_depth_mm: 2 (0.58%)
  - flipper_length_mm: 2 (0.58%)
  - body_mass_g: 2 (0.58%)
  - sex: 11 (3.2%)

## Descripción


## Hipótesis

- **H1**: flipper_length_mm se asocia positivamente con body_mass_g
  - Tests sugeridos: pearson_correlation, spearman_correlation
- **H2**: bill_length_mm difiere significativamente entre species
  - Tests sugeridos: kruskal_wallis, one_way_anova
- **H3**: La distribución de species se asocia con island
  - Tests sugeridos: chi_squared
- **H4**: body_mass_g difiere entre sexos (Male vs Female)
  - Tests sugeridos: mann_whitney_u, t_test_independent

## Pruebas estadísticas

- **H1** (pearson_correlation): stat=0.8712, p=4.37e-107 → Apoya hipótesis
- **H1** (spearman_correlation): stat=0.84, p=2.76e-92 → Apoya hipótesis
- **H2** (kruskal_wallis): stat=244.1367, p=9.69e-54 → Apoya hipótesis
- **H3** (chi_squared): stat=299.5503, p=1.35e-63 → Apoya hipótesis
- **H4** (mann_whitney_u): stat=20845.5, p=1.81e-15 → Apoya hipótesis

## Conclusiones


### Hallazgos descriptivos
- El dataset contiene 344 observaciones de 3 especies de pingüinos con 7 variables (refs: 00_raw_profile.json:profile.n_rows, 00_raw_profile.json:profile.n_cols)
- Existen valores faltantes concentrados en sex y variables de medición corporal (refs: 00_raw_profile.json:missing)
- Las variables numéricas presentan rangos y dispersiones diferentes según especie (refs: 04_descriptive_stats.json:numeric_summary)

### Patrones visuales
- Los histogramas de bill_length_mm muestran distribuciones diferenciadas por especie (refs: 05_visual_registry.json:hist_bill_length_mm_by_species)
- El scatter de body_mass_g vs flipper_length_mm muestra agrupaciones por especie (refs: 05_visual_registry.json:scatter_body_mass_g_vs_flipper_length_mm_hue_species)

## Gráficos generados

- [count_species](artifacts\plots\count_species.png)
- [count_island](artifacts\plots\count_island.png)
- [count_sex](artifacts\plots\count_sex.png)
- [count_island_by_species](artifacts\plots\count_island_by_species.png)
- [count_sex_by_species](artifacts\plots\count_sex_by_species.png)
- [hist_body_mass_g_by_species](artifacts\plots\hist_body_mass_g_by_species.png)
- [hist_flipper_length_mm_by_species](artifacts\plots\hist_flipper_length_mm_by_species.png)
- [hist_bill_length_mm_by_species](artifacts\plots\hist_bill_length_mm_by_species.png)
- [hist_bill_depth_mm_by_species](artifacts\plots\hist_bill_depth_mm_by_species.png)
- [box_body_mass_g_by_species](artifacts\plots\box_body_mass_g_by_species.png)
- [box_flipper_length_mm_by_species](artifacts\plots\box_flipper_length_mm_by_species.png)
- [box_bill_length_mm_by_species](artifacts\plots\box_bill_length_mm_by_species.png)
- [scatter_body_mass_g_vs_flipper_length_mm_hue_species](artifacts\plots\scatter_body_mass_g_vs_flipper_length_mm_hue_species.png)
- [heatmap_corr_pearson](artifacts\plots\heatmap_corr_pearson.png)
- [heatmap_corr_spearman](artifacts\plots\heatmap_corr_spearman.png)