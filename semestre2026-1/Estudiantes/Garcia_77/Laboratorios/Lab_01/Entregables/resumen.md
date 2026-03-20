# Resumen Ejecutivo - Lab_01

## Objetivo
Explorar el dataset de penguins, formular hipotesis falsables y evaluar evidencia estadistica reproducible.

## Datos
- 344 filas y 7 columnas.
- Faltantes globales: 0.79%.
- Sin filas duplicadas.

## Hallazgos clave
- Asociacion fuerte entre flipper_length_mm y body_mass_g.
- Diferencias de bill_length_mm entre especies.
- Asociacion entre species e island.

## Evidencia estadistica (alpha = 0.05)
- H1 Pearson: p = 4.3706809630006207e-107 -> rechazar_H0.
- H1 Spearman: p = 2.763218997179663e-92 -> rechazar_H0.
- H2 ANOVA: p = 2.6946137388895146e-91 -> rechazar_H0.
- H2 Kruskal-Wallis: p = 9.691371997194331e-54 -> rechazar_H0.
- H3 Chi-cuadrado: p = 1.354573829719252e-63 -> rechazar_H0.

## Proximos pasos
1. Evaluar correlaciones por especie para flipper_length_mm vs body_mass_g.
2. Profundizar en el papel de sex, especialmente por faltantes.
3. Extender comparaciones a nuevos datasets para validar generalizacion del enfoque.
