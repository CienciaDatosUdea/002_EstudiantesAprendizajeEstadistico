# Reproducible Data Observatory Report

## Descriptive Findings
- The dataset contains 344 rows and 7 columns (ref: `00_raw_profile.json:rows`, `00_raw_profile.json:columns`).
- Category coverage includes 3 species, 3 islands, and 2 reported sex categories, with 11 missing entries in sex (ref: `04_descriptive_stats.json:categorical_summary.species.nunique`, `04_descriptive_stats.json:categorical_summary.island.nunique`, `04_descriptive_stats.json:categorical_summary.sex.nunique`, `04_descriptive_stats.json:categorical_summary.sex.top_values.nan`).
- Missing values appear in each measured numeric variable with 2 missing rows per variable (ref: `00_raw_profile.json:missing_report.bill_length_mm.missing_count`, `00_raw_profile.json:missing_report.bill_depth_mm.missing_count`, `00_raw_profile.json:missing_report.flipper_length_mm.missing_count`, `00_raw_profile.json:missing_report.body_mass_g.missing_count`).
- No duplicate rows were reported (ref: `00_raw_profile.json:duplicates_report.duplicate_rows`).

## Visual Patterns
- Species-level differences are visible for body mass in grouped histograms and boxplots (ref: `05_visual_registry.json:hist_07`, `05_visual_registry.json:box_08`).
- Species-level differences are visible for flipper length in grouped histograms and boxplots (ref: `05_visual_registry.json:hist_06`, `05_visual_registry.json:box_09`).
- The correlation heatmap pattern is consistent with a positive association between flipper length and body mass, and a negative association between bill depth and flipper length (ref: `05_visual_registry.json:heatmap_10`, `04_descriptive_stats.json:correlation_matrices.pearson.flipper_length_mm.body_mass_g`, `04_descriptive_stats.json:correlation_matrices.pearson.bill_depth_mm.flipper_length_mm`).
- Species composition appears uneven across island categories (ref: `04_descriptive_stats.json:crosstabs.species_x_island`, `05_visual_registry.json:count_01`, `05_visual_registry.json:count_02`).

## Next Hypotheses To Test
- Test whether mean body mass differs by species using Levene + ANOVA/Kruskal (ref: `05_visual_registry.json:box_08`, `05_visual_registry.json:hist_07`).
- Test whether mean flipper length differs by species using Levene + ANOVA/Kruskal (ref: `05_visual_registry.json:box_09`, `05_visual_registry.json:hist_06`).
- Test association between species and island using chi-squared independence (ref: `04_descriptive_stats.json:crosstabs.species_x_island`).
- Confirm monotonic association between flipper length and body mass using Spearman (ref: `04_descriptive_stats.json:correlation_matrices.spearman.flipper_length_mm.body_mass_g`).
