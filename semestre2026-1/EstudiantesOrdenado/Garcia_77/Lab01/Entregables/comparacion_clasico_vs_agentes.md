# Comparacion: Enfoque Clasico vs Enfoque con Agentes

## 1) Estructura de trabajo

### Enfoque clasico
- Notebook unico con exploracion, visualizacion, hipotesis, pruebas y conclusiones en celdas.
- Ventaja: iteracion rapida y visual inmediata.
- Riesgo: menor control de trazabilidad si no se documenta paso a paso.

### Enfoque con agentes
- Separacion explicita entre planificacion (agente) y ejecucion (runner).
- Resultados persistidos en artifacts JSON/PNG y auditados en run_log.
- Ventaja: alta reproducibilidad y mejor control anti-hallucination.

## 2) Reproducibilidad

### Clasico
- Depende de orden de ejecucion de celdas y estado del kernel.
- Puede requerir mas disciplina manual para mantener coherencia.

### Agentes
- Cada operacion genera artifact con timestamp.
- run_log permite reconstruir exactamente que se ejecuto, cuando y con que parametros.

## 3) Calidad de evidencia

### Clasico
- Facil para explorar y ajustar en tiempo real.
- Puede mezclar observacion subjetiva con resultados sin trazabilidad fuerte.

### Agentes
- Las afirmaciones se anclan a artifacts concretos.
- Menor probabilidad de afirmar algo sin respaldo verificable.

## 4) Carga de trabajo del estudiante

### Clasico
- Menor overhead inicial.
- Mayor probabilidad de perder registro formal del proceso.

### Agentes
- Mayor overhead inicial (prompts, fase, artifacts).
- Mejor entrenamiento en metodo cientifico: observar, describir, hipotetizar y contrastar.

## 5) Resultado en este Lab_01
- Ambos enfoques convergieron en hallazgos consistentes:
  - Asociacion fuerte flipper_length_mm-body_mass_g.
  - Diferencias de bill_length_mm entre especies.
  - Asociacion species-island.
- El enfoque con agentes dejo evidencia mas estructurada para auditoria y entrega final.

## Conclusión
Para aprendizaje exploratorio rapido, el enfoque clasico es util. Para evaluacion reproducible, trazabilidad y comunicacion formal de resultados, el enfoque con agentes es superior en este laboratorio.
