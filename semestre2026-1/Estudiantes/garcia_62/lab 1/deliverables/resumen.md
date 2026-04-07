# Resumen - Fase 2 (Enfoque con Agentes)

## Arquitectura
- Agente: planifica, interpreta y redacta salida con evidencia.
- Runner: ejecuta codigo Python, calcula estadisticas, genera graficos y escribe artifacts.
- Artifacts: fuente unica de evidencia verificable.

## Flujo aplicado
1. OBSERVE: generacion de perfil del dataset en artifacts/00_raw_profile.json.
2. DESCRIBE: estadistica descriptiva y visuales en artifacts/04_descriptive_stats.json y artifacts/05_visual_registry.json.
3. HYPOTHESIZE_AND_CONCLUDE: hipotesis y conclusiones en artifacts/06_hypotheses_log.json y artifacts/07_conclusions.json.

## Principios cumplidos
- Sin invencion de valores.
- Evidencia trazable via evidence_refs.
- Separacion clara entre planificacion (agente) y computo (runner).
- Reproducibilidad por artifacts en disco.

## Salidas principales
- artifacts/report.md
- artifacts/report.html
- artifacts/06_hypotheses_log.json
- artifacts/07_conclusions.json
