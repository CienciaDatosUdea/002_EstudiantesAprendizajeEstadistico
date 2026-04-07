# Comparación de enfoques: Enfoque Clásico vs Enfoque con Agentes

Documento técnico que compara metodologías aplicadas en el Lab_01_pinguinos.
El análisis es académico y se apoya en los artifacts generados durante la Fase 2.

---

## 1. Metodología

- Enfoque Clásico: el investigador (Victor Palacios) controla iterativamente el flujo de trabajo en un entorno interactivo (p. ej. notebooks). El proceso es altamente flexible y permite exploración ad hoc, pero depende del juicio humano en cada paso (selección de variables, transformaciones, pruebas). Esto facilita experimentación rápida, pero introduce riesgo de sesgos no documentados, decisiones implícitas y dificultades para reproducir exactamente la secuencia de acciones.

- Enfoque con Agentes: se aplica el principio de 'Separación de Preocupaciones'. El agente actúa como planificador (elige operaciones, genera planes JSON de alto nivel) y el Runner ejecuta las operaciones (código reproducible que produce artifacts estandarizados: JSON y PNG). Esta separación clarifica responsabilidades: el agente decide qué pedir y por qué; el Runner realiza cálculos y retorna evidencia estructurada.

Ventaja clave: la separación reduce decisiones ad hoc no registradas y obliga a que cada conclusión referencie un artifact (traceability).

---

## 2. Rigor y verificabilidad

Artifact: en este contexto, un "artifact" es un fichero estructurado (normalmente JSON) o una visualización (PNG) que contiene la salida completa de una operación computacional —por ejemplo, `profile_dataset`, `numeric_summary`, `correlation_matrix`— y que puede almacenarse, inspeccionarse y volver a evaluarse.

- En la Fase 2, ninguna conclusión final fue emitida sin una referencia explícita a artifacts (`evidence_refs`). Ejemplos: `FASE_2_AGENTES_profile_dataset_20260402_130023.json`, `FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json`, `FASE_2_AGENTES_box_body_mass_g_by_species_20260402_130047.png`.

- Las Reglas Anti-Alucinación implementadas (no inventar números, citar evidencia, evitar lenguaje causal sin soporte) garantizan que todas las cifras y veredictos provengan de artifacts reproducibles. Esto mejora la verificabilidad: cualquier tercero puede abrir los JSON/PNG y comprobar los valores citados.

---

## 3. Reproducibilidad y generalización

- Enfoque Clásico: reproducir exactamente un análisis manual requiere conservar el notebook, versiones de librerías y el estado interactivo. Pequeñas diferencias en orden de ejecución o parámetros pueden producir resultados distintos.

- Enfoque con Agentes: la ejecución se basa en scripts (Runner) y planes formales (JSON). Si se conservan los artifacts y el Runner, repetir el análisis es una operación determinista: ejecutar las mismas operaciones produce los mismos artifacts, facilitando auditoría y reproducibilidad.

- Experimento de generalización: el Runner fue probado con otro dataset (por ejemplo `iris`) sin cambios en la lógica de procesamiento (solo adaptando los nombres de columnas en el plan). Esto demuestra que la arquitectura se adapta a distintos esquemas de datos siempre que las operaciones solicitadas sean genéricas (perfilado, resúmenes numéricos/categóricos, correlaciones, gráficos). La capacidad de reutilizar el Runner favorece la escalabilidad y la reproducibilidad entre dominios.

---

## 4. Tabla comparativa

| Criterio | Enfoque Clásico | Enfoque con Agentes |
|---|---:|---:|
| Velocidad (prototipado) | Alta en exploración manual; el usuario puede iterar rápido en un notebook. | Inicialmente más lento por la definición de planes y scripts, pero acelerado para re-ejecuciones y pipelines. |
| Trazabilidad | Baja a media: requiere documentación manual y buenas prácticas para auditar pasos. | Alta: cada salida es un artifact con nombre y timestamp; decisiones referencian `evidence_refs`. |
| Riesgo de Error | Mayor riesgo de errores humanos y decisiones implícitas no registradas. | Menor riesgo de errores de procedimiento; sin embargo, depende de la correcta implementación del Runner y del plan del agente. |
| Escalabilidad | Limitada: replicar a múltiples datasets o a producción exige adaptar notebooks manualmente. | Alta: planes y Runner permiten automatizar y paralelizar análisis sobre múltiples datasets (reutilización más sencilla). |
| Reproducibilidad | Moderada: reproducible si se conserva entorno y notebook, pero frágil ante cambios. | Alta: outputs deterministas si se reutilizan los mismos artifacts y runner; facilita auditoría. |

---

## 5. Reflexión final de Palacios_10

Desde la perspectiva metodológica y aplicando criterios de reproducibilidad y transparencia, el "Enfoque con Agentes" representa hoy un estándar moderno para observatorios de datos reproducibles. Sus ventajas principales son:

- Documentación automática y obligatoria de resultados (artifacts), que reduce la posibilidad de conclusiones sin evidencia.
- Separación de responsabilidades que facilita revisión independiente: el plan es legible y el Runner produce evidencia inmutable.
- Escalabilidad y generalización: el mismo Runner puede aplicarse a diferentes datasets sin reescribir la lógica analítica.

No obstante, el enfoque agéntico no reemplaza la creatividad y juicio del analista; la recomendación práctica es combinar ambos: usar notebooks para exploración y prototipado rápido, y el enfoque con agentes para producción, auditoría y entrega reproducible.

---

## 6. Referencias en el repositorio

- Fase 1 (clásica): `FASE_1_CLASICO/Analisis_Clasico_Pinguinos.ipynb`
- Fase 2 (agentes): `FASE_2_AGENTES/student_log.md`, `FASE_2_AGENTES/report.md`, `FASE_2_AGENTES/artifacts/` (JSON y `plots/`)

---

## 7. Apéndice empírico: métricas y artifacts generados

Los siguientes recuentos y listados se obtuvieron directamente del directorio de artifacts producido por la Fase 2 (`FASE_2_AGENTES/artifacts/`) y su subdirectorio `plots/`. Las cifras se listan para documentar la producción científica del pipeline y facilitar auditoría.

- Número total de artifacts JSON en `FASE_2_AGENTES/artifacts/`: 20
- Número total de imágenes PNG en `FASE_2_AGENTES/artifacts/plots/`: 12

Listado representativo de JSON (ejemplos con timestamps):

- `06_hypotheses_log.json`
- `07_conclusions.json`
- `FASE_2_AGENTES_profile_dataset_20260402_134637.json`
- `FASE_2_AGENTES_missing_report_20260402_134637.json`
- `FASE_2_AGENTES_numeric_summary_20260402_134637.json`
- `FASE_2_AGENTES_correlation_matrix_pearson_20260402_134638.json`
- `FASE_2_AGENTES_crosstab_20260402_134638.json`

Listado de PNG en `FASE_2_AGENTES/artifacts/plots/` (selección):

- `FASE_2_AGENTES_box_body_mass_g_by_species_20260402_134701.png`
- `FASE_2_AGENTES_scatter_bill_length_mm_vs_bill_depth_mm_hue_species_20260402_134711.png`
- `FASE_2_AGENTES_hist_body_mass_g_20260402_134658.png`
- `FASE_2_AGENTES_heatmap_corr_pearson_20260402_134714.png`

Estas listas permiten que un revisor abra los archivos exactos y verifique los valores citados en este documento.

---

## 8. Reproducibilidad práctica — comandos útiles

Para reproducir o regenerar los artifacts y los entregables en este repositorio, ejecuciones típicas:

```bash
# Ejecutar el Runner para una fase (ejemplo: DESCRIBE)
cd LAB1_PINGUINOS/FASE_2_AGENTES
python runner.py

# Generar los entregables compilados
cd LAB1_PINGUINOS
python generar_entrega.py
```

Nota: conservar los artifacts JSON y las versiones de librerías garantiza que la regeneración produzca resultados idénticos.

---

## 9. Estilo y presentación (recomendaciones académicas)

- Incluir un título, autor, fecha y resumen ejecutivo en todos los entregables (ya presente en `report.md`).
- En `reporte_final.html` incorporar figuras con captions y referencias cruzadas a los JSON que contienen los datos numéricos.
- Añadir una sección de "Limitaciones" explícita (tamaños de muestra por especie, missingness en `sex`) y anotar cómo se trataron estas limitaciones.
- Incluir un apéndice con los comandos exactos y hashes de commit (si aplica) para trazabilidad completa.

---


