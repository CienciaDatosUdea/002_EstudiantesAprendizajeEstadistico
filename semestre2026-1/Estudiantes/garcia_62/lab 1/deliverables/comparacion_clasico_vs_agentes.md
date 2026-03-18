# Comparacion: Enfoque Clasico vs Enfoque con Agentes

## Objetivo
Comparar calidad analitica, reproducibilidad y trazabilidad entre ambos enfoques.

## Tabla comparativa
| Criterio | Enfoque Clasico | Enfoque con Agentes |
|---|---|---|
| Separacion planificacion/ejecucion | Baja o mixta | Alta (agente vs runner) |
| Reproducibilidad | Depende del notebook y notas | Alta por artifacts versionables |
| Trazabilidad de evidencia | Variable | Explicita via evidence_refs |
| Riesgo de invencion de resultados | Medio | Menor con reglas anti-alucinacion |
| Escalabilidad a nuevos datasets | Manual | Estructurada por fases |
| Costo cognitivo inicial | Menor | Mayor al inicio |
| Mantenimiento del flujo | Heterogeneo | Estandarizado |

## Observaciones de esta implementacion
- El flujo por fases permitio separar claramente OBSERVE, DESCRIBE y HYPOTHESIZE_AND_CONCLUDE.
- La interpretacion final se redacto solo con evidencia guardada en artifacts.
- El enfoque facilito auditar de donde sale cada afirmacion.

## Pendientes para comparacion completa
1. Ejecutar un flujo clasico equivalente con el mismo dataset.
2. Medir tiempos y numero de iteraciones en ambos enfoques.
3. Comparar claridad de conclusiones en una rubrica comun.
4. Repetir con un segundo dataset para evaluar generalizacion.
