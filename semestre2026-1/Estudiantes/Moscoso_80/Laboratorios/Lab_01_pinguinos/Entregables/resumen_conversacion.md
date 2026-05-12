# Resumen de Conversación y Evaluación Crítica
## Lab_01_pinguinos · Moscoso_80 · 2026-03-09

> Este documento resume el proceso completo de análisis del laboratorio, identifica
> aciertos y errores en el razonamiento del estudiante, y sirve como registro reflexivo
> del aprendizaje a través del flujo Agente–Runner.

---

## Estructura general del flujo

```
Phase 0 — STUDENT_PREANALYSIS   (sin código, solo razonamiento previo)
Phase 1 — OBSERVE               (runner genera 4 artifacts de estructura)
Phase 2 — DESCRIBE              (runner genera estadísticas y 11 plots)
Phase 3 — HYPOTHESIZE           (runner genera 2 crosstabs y 2 plots adicionales)
```

Incidentes técnicos registrados:
- **Bug 1 (runner):** `np.fill_diagonal(corr_abs.values, 0)` fallaba en numpy moderno (array de solo lectura). Solución: `.to_numpy().copy()`.
- **Bug 2 (runner):** `select_dtypes(include=["object"])` generaba `Pandas4Warning`. Solución: reemplazar por `"str"`.
- **Mejora al runner:** se agregó soporte para `--phase HYPOTHESIZE --ops "op(args)"` para ejecutar operaciones individuales sin repetir el pipeline completo de DESCRIBE.

---

## Phase 0 — Pre-análisis del estudiante

### Lo que escribiste

- Cada fila representa un pingüino individual con datos morfológicos, isla y sexo.
- Intuiste que diferentes especies tendrán morfologías distintas por adaptación al entorno (analogía con picos de aves en islas).
- Esperabas correlación entre `sex` y `body_mass_g`, y entre `flipper_length_mm` y `body_mass_g`.
- Anticipaste que una isla podría ser compartida por dos o más especies.

### Evaluación crítica

| Razonamiento | Evaluación |
|---|---|
| Analogía con adaptación por isla | ✅ **Acertado como marco conceptual.** Válido para motivar hipótesis, aunque no debía usarse como conclusión sin evidencia. |
| Correlación `flipper_length_mm` × `body_mass_g` | ✅ **Acertado.** Anticipaste la relación más fuerte del dataset (r=0.871). Estimaste 0.6–0.7, que fue conservador pero en la dirección correcta. |
| Correlación `sex` × `body_mass_g` | ✅ **Razonable.** Hipótesis plausible, aunque su verificación requiere controlar por especie (H5 pendiente). |
| "No creo que las islas sean exclusivas de una especie" | ⚠️ **Parcialmente errado.** Adelie sí comparte las 3 islas, pero Gentoo y Chinstrap son completamente exclusivas de una. La estructura es más rígida de lo que anticipaste. |

**Calidad general del Phase 0:** Alta. Las preguntas fueron respondidas con razonamiento, no con intuición vacía. La distinción entre lo que es observable (columnas, tipos) y lo que es especulación fue correctamente mantenida.

---

## Phase 1 — OBSERVE

### Hallazgos del runner

- 344 filas, 7 columnas, 0 duplicados.
- 19 faltantes totales (0.79%): `sex` concentra 11 (3.2%), las 4 numéricas tienen 2 cada una (0.58%).

### Evaluación crítica de tus respuestas

**Pregunta: ¿Los faltantes en `sex` son aleatorios?**

> Respuesta del estudiante: "Podrían ser aleatorios. Con 11/344 no es crítico. Podríamos eliminar o imputar."

✅ **Correcto en la actitud metodológica** — proponer eliminar vs imputar según el mecanismo de faltantes (MCAR/MAR/MNAR) es el enfoque correcto.  
⚠️ **Incompleto:** no propusiste cómo *verificar* si son aleatorios (ej. crosstab de `sex_missing` vs `species`). Identificar el mecanismo de faltantes antes de decidir la estrategia es un paso que faltó.

**Pregunta: ¿Las 2 filas con numéricas faltantes son las mismas?**

> Respuesta: "Sí, probablemente. Se puede ver con una máscara."

✅ **Acertado** — la respuesta es correcta y la herramienta propuesta (`df[df.isnull().any(axis=1)]`) es exactamente la correcta.

---

## Phase 2 — DESCRIBE

### Evaluación crítica de tus respuestas

**Pregunta sobre la correlación negativa global `bill_depth_mm` × `bill_length_mm` (-0.235):**

> Respuesta: "Para ver si el Pearson de -0.235 es consistente, deberíamos hacer un scatter de bill_depth y bill_length."

✅ **Excelente.** Identificaste exactamente que la correlación global no puede interpretarse sin separar por especie, y propusiste el gráfico correcto para verificarlo. Esto es la intuición de la Paradoja de Simpson — llegaste a ella antes de verla nombrada.

**Pregunta sobre el desbalance de especies y la prueba más robusta:**

> Respuesta: "Se me ocurre aplicar Bootstrapping agrupado por especies."

⚠️ **Parcialmente correcto pero innecesariamente complejo.** Bootstrapping es válido, pero para comparar `body_mass_g` entre grupos con tamaños distintos, las pruebas no paramétricas estándar (Kruskal-Wallis, Dunn post-hoc) son más directas, están diseñadas para este caso y no requieren supuestos de normalidad. Bootstrapping es más útil cuando necesitas construir intervalos de confianza para estimadores personalizados.

**Pregunta sobre la correlación `flipper_length_mm` × `body_mass_g` = 0.871:**

> Respuesta: "Esperaba 0.6-0.7. No me sorprende porque un pingüino más robusto puede tener aletas más largas. Esperaría ver una recta con pendiente alta pero con casos de pingüinos pequeños con gran masa (pequeños y gordos)."

✅ **Acertado en el razonamiento cualitativo.** La imagen mental de "pequeños y gordos" como fuente de dispersión es correcta — la correlación no es perfecta por exactamente esa razón.

---

## Phase 3 — HYPOTHESIZE

### Evaluación crítica de tus hipótesis

**H_estudiante_1: bill_length + flipper_length separan bien las tres especies (reduccionismo en 3 regiones del scatter)**

✅ **Excelente hipótesis y bien razonada.** La formulación en términos de combinaciones de variables ("bill_length bajo + flipper corto → Adelie") es prácticamente una descripción verbal de un clasificador de árbol de decisión de dos variables. Fue el razonamiento más sofisticado del laboratorio.

**H_estudiante_2: El skewness de `body_mass_g` se explica por mayor proporción de machos en Gentoo**

⚠️ **Hipótesis razonable pero falsada por los datos.** El crosstab mostró solo 3 machos adicionales en Gentoo (61 vs 58), insuficiente para explicar un skewness de 0.47. El crédito está en que formulaste la hipótesis de forma que podía ser refutada — eso es pensamiento científico correcto. El error fue en la magnitud esperada del efecto.

**H_estudiante_3: "Se siente extraño plantear una correlación entre dos variables solo porque tienen proporciones similares"**

✅ **Reflexión metodológica muy valiosa.** Identificaste correctamente que comparar proporciones marginales no es suficiente para establecer una relación entre variables. El crosstab (tabla de contingencia conjunta) es exactamente la herramienta que faltaba — y lo reconociste.

---

## Resumen de aciertos y áreas de mejora

### ✅ Aciertos destacados

1. **Paradoja de Simpson:** identificada intuitivamente antes de ver el scatter, solo desde el coeficiente global.
2. **Hipótesis falsables:** todas las hipótesis propias tenían variables concretas y podían ser refutadas.
3. **Pensamiento proporcional:** reconociste que comparar porcentajes marginales no es lo mismo que analizar la relación conjunta entre variables.
4. **Estimación razonada:** predijiste la correlación `flipper` × `mass` en 0.6–0.7 antes de verla, con justificación biológica.
5. **Autocorrección:** cuando la hipótesis sobre machos de Gentoo fue refutada por los datos, la aceptaste sin resistencia.

### ⚠️ Áreas de mejora

1. **Mecanismo de faltantes:** proponer eliminar vs imputar sin primero verificar si el patrón de faltantes es aleatorio (test de Little, o al menos un crosstab `sex_missing` × `species`).
2. **Elección de prueba estadística:** para comparar grupos de tamaños distintos, preferir Kruskal-Wallis sobre Bootstrapping como primera opción — es más directo y menos costoso computacionalmente.
3. **Cuantificar expectativas:** en Phase 0 las expectativas son cualitativas ("esperaba una correlación"). Entrenar el hábito de dar un rango numérico esperado hace que la comparación con los resultados sea más informativa.
4. **Control de variables de confusión:** varias hipótesis sobre `sex` × `body_mass_g` debían especificar desde el inicio "controlando por especie" — el efecto de confusión de `species` es evidente en los datos.

---

## Línea de tiempo de la conversación

| Fase | Momento clave | Resultado |
|---|---|---|
| Phase 0 | Pre-análisis correcto con analogía evolutiva | Hipótesis bien formuladas |
| Phase 1 | Artifacts generados correctamente | 0 duplicados, patrón de faltantes identificado |
| Phase 2 | Runner crashea (`ValueError: read-only array`) | Bug corregido; 11 plots generados |
| Phase 2 | Paradoja de Simpson identificada por el estudiante | Solicitud de scatter adicional |
| Phase 3 | Runner no soportaba operaciones individuales | Mejora: `--phase HYPOTHESIZE --ops` implementado |
| Phase 3 | Crosstab species×sex refuta hipótesis del skewness | Hipótesis correctamente descartada |
| Final | 3 archivos de entrega creados | Reporte completo y reproducible |
