# Resumen ejecutivo — Lab_01_pinguinos

**Autor:** Palacios_10

**Fecha:** 2026-04-02

---

## Introducción

Este documento resume el trabajo realizado en el Observatorio de Datos sobre el dataset "Palmer Penguins" (versión utilizada en el laboratorio). Objetivo: caracterizar morfológicamente las tres especies (`Adelie`, `Gentoo`, `Chinstrap`) y evaluar hipótesis sobre masa corporal, relaciones morfológicas y calidad de los datos.

- Tamaño del dataset: **344 registros**.
- Variables consideradas: **7** (`species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`).

Fuente de la información estructural: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_profile_dataset_20260402_134637.json`.

---

## Hipótesis documentadas (Fase 3)

Se trabajaron cuatro hipótesis falsables. Para cada una se indica el enunciado, el resultado y la evidencia técnica (archivos generados).

### H1 — Masa corporal por especie
- **Enunciado:** La mediana de `body_mass_g` es mayor en la especie `Gentoo` que en `Adelie` y `Chinstrap`.
- **Resultado:** **Apoyada** (visualmente y por resumen categórico).
- **Evidencia técnica:**
  - `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_box_body_mass_g_by_species_20260402_134701.png` (boxplot por especie)
  - `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_categorical_summary_20260402_134637.json`
  - Registro de hipótesis: `FASE_2_AGENTES/artifacts/06_hypotheses_log.json`

### H2 — Correlación aleta ↔ masa corporal
- **Enunciado:** Existe una asociación positiva entre `flipper_length_mm` y `body_mass_g`.
- **Resultado:** **Apoyada (global)** — asociación fuerte a nivel agregado; falta desagregación por especie para confirmar internamente.
- **Evidencia técnica:**
  - `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json` (matriz de correlación, r ≈ 0.8712 entre aleta y masa)
  - `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_heatmap_corr_pearson_20260402_134714.png` (heatmap)
  - Registro de hipótesis: `FASE_2_AGENTES/artifacts/06_hypotheses_log.json`

### H3 — Diferencias en medidas del pico entre especies
- **Enunciado:** Las tres especies difieren en la distribución conjunta de `bill_length_mm` y `bill_depth_mm`.
- **Resultado:** **Apoyada** (visualmente por separación en el espacio de pico).
- **Evidencia técnica:**
  - `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_scatter_bill_length_mm_vs_bill_depth_mm_hue_species_20260402_134711.png`
  - Registro de conclusiones: `FASE_2_AGENTES/artifacts/07_conclusions.json`

### H4 — Missingness de `sex` respecto a `species`
- **Enunciado:** La probabilidad de que `sex` esté ausente es independiente de la `species` (MCAR frente a `species`).
- **Resultado:** **Inconclusa** — se detectaron faltantes pero no existe artefacto conclusivo para probar independencia.
- **Evidencia técnica:**
  - `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json` (11 faltantes en `sex`)
  - `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_crosstab_20260402_134638.json` (tabla `species` vs `island`, no suficiente para MCAR sobre `sex`)
  - Registro de hipótesis: `FASE_2_AGENTES/artifacts/06_hypotheses_log.json`

---

## Conclusiones (tres capas)

### Capa 1 — Hallazgos descriptivos (hechos verificados)
- El dataset contiene **344** observaciones y **7** variables (ver `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_profile_dataset_20260402_134637.json`).
- No se detectaron filas duplicadas (`FASE_2_AGENTES/artifacts/FASE_2_AGENTES_duplicates_report_20260402_134637.json` muestra `n_duplicates: 0`).
- Valores faltantes: 2 faltantes en cada variable numérica (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) y 11 faltantes en `sex` (`FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json`).

### Capa 2 — Patrones visuales y asociaciones
- Los **Gentoo** presentan una mediana de masa corporal superior con menor solapamiento de IQR frente a `Adelie` y `Chinstrap` (ver `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_box_body_mass_g_by_species_20260402_134701.png`).
- La asociación entre `flipper_length_mm` y `body_mass_g` es fuerte a nivel agregado (r≈0.87 en `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json`), lo que sugiere que la longitud de la aleta es un predictor potente de masa corporal en este conjunto de datos.
- Las medidas del pico (`bill_length_mm`, `bill_depth_mm`) muestran clusters por especie en el scatter coloreado por especie (`FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_scatter_bill_length_mm_vs_bill_depth_mm_hue_species_20260402_134711.png`).

### Capa 3 — Próximas hipótesis y trabajos sugeridos
- Verificar H1 con ANOVA o Kruskal-Wallis y post-hoc; almacenar `FASE_3_h1_tests_<ts>.json`.
- Calcular correlaciones por especie entre `flipper_length_mm` y `body_mass_g` (`FASE_3_corr_by_species_<ts>.json`).
- Realizar MANOVA o pruebas univariadas para confirmar diferencias multivariadas del pico (`FASE_3_h3_tests_<ts>.json`).
- Investigar `sex` missingness con `species` × `sex_missing` y aplicar chi-cuadrado o Fisher; guardar `FASE_3_missing_sex_test_<ts>.json`.
- Explorar factores ambientales (año, isla, clima) si están disponibles para estudiar su impacto sobre masa corporal y variación morfológica.

---

## Preguntas de investigación para el investigador

1. **Tratamiento de `sex`:** ¿Preferirá el equipo imputar `sex` por especie (modo/mediana condicional), marcar como `Unknown`, o excluir registros sin `sex` en análisis estratificados? Referencia: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json`.
2. **Exclusividad por isla:** ¿Desea que se pruebe formalmente si ciertas especies son exclusivas o dominantes por isla (usar `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_crosstab_20260402_134638.json`)?

---

## Rigor técnico

El análisis siguió un flujo reproducible y exigente en evidencia:

- **Arquitectura agéntica:** El Agente generó planes y el Runner ejecutó las operaciones, produciendo artifacts con timestamp en `FASE_2_AGENTES/artifacts/`.
- **Reglas anti-alucinación:** Ninguna conclusión final fue emitida sin una referencia explícita a artifacts (`evidence_refs`). Ver `FASE_2_AGENTES/artifacts/06_hypotheses_log.json` y `FASE_2_AGENTES/artifacts/07_conclusions.json`.
- **Reproducibilidad práctica:** El script `LAB1_PINGUINOS/generar_entrega.py` compila los entregables en `LAB1_PINGUINOS/Entregables/` y permite regenerar reportes si se conservan los artifacts y el entorno.

---

## Artefacts clave

- Perfil y calidad: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_profile_dataset_20260402_134637.json`, `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_missing_report_20260402_134637.json`.
- Estadística y correlaciones: `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_numeric_summary_20260402_134637.json`, `FASE_2_AGENTES/artifacts/FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json`.
- Hipótesis y conclusiones: `FASE_2_AGENTES/artifacts/06_hypotheses_log.json`, `FASE_2_AGENTES/artifacts/07_conclusions.json`.
- Gráficos representativos: `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_box_body_mass_g_by_species_20260402_134701.png`, `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_scatter_bill_length_mm_vs_bill_depth_mm_hue_species_20260402_134711.png`, `FASE_2_AGENTES/artifacts/plots/FASE_2_AGENTES_heatmap_corr_pearson_20260402_134714.png`.

---


