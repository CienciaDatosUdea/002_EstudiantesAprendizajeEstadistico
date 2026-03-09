# Comparación de Enfoques: Clásico vs Agentes
## Lab_01_pinguinos · Moscoso_80 · 2026-03-09

> Comparación entre el enfoque clásico (notebook con código manual) y el enfoque con
> agentes (runner + artifacts + conversación socrática). El criterio de evaluación
> es pedagógico: ¿qué enfoque desarrolla mejor la capacidad de análisis crítico?

---

## 1 — Descripción de cada enfoque

### Enfoque Clásico (Fase 1 del laboratorio)

El estudiante escribe código directamente en el notebook para responder preguntas
predefinidas (Partes A–F). El flujo es:

```
Pregunta del laboratorio → El estudiante escribe código → Ve el output → Redacta respuesta
```

El análisis está guiado por las preguntas del enunciado. El estudiante decide
cómo responderlas, pero el *qué preguntar* ya está dado.

### Enfoque con Agentes (Fase 2 del laboratorio)

El estudiante interactúa con un agente planificador (Claude) que propone qué operaciones
ejecutar y por qué. El runner ejecuta el código y escribe artifacts. El agente lee
los artifacts y propone preguntas socráticas. El flujo es:

```
Agente propone → Runner ejecuta → Artifacts escritos → Agente interpreta
→ Preguntas socráticas → Estudiante razona → Siguiente fase
```

El *qué preguntar* emerge del análisis previo, no de un enunciado fijo.

---

## 2 — Comparación por dimensión

### 2.1 — Escritura de código

| Dimensión | Clásico | Con Agentes |
|---|---|---|
| ¿Quién escribe el código? | El estudiante | El runner (automatizado) |
| ¿El estudiante entiende el código? | Sí, lo construye línea a línea | Parcialmente — puede revisar `runner.py` |
| ¿El código es reproducible? | Depende de si el estudiante lo organiza bien | Sí, siempre: timestamps, rutas fijas, sin sobrescritura |
| ¿Hay riesgo de error de implementación? | Alto (typos, lógica incorrecta) | Bajo (el runner está validado una sola vez) |
| ¿Se aprende a programar? | Más | Menos |

**Análisis:** el enfoque clásico es mejor para aprender a programar. El enfoque con
agentes es mejor para aprender a *usar* código como herramienta de análisis sin que
la implementación distraiga del razonamiento estadístico.

---

### 2.2 — Razonamiento analítico

| Dimensión | Clásico | Con Agentes |
|---|---|---|
| ¿El estudiante propone hipótesis propias? | Después de ver los datos (Parte D) | Antes de ver los datos (Phase 0) y en cada fase |
| ¿Las hipótesis son falsables? | Depende del estudiante | El agente exige que lo sean explícitamente |
| ¿Se contrasta expectativa vs resultado? | Raramente | Siempre (Phase 0 se compara con Phase 3) |
| ¿El estudiante explica inconsistencias? | Solo si el enunciado lo pide | Obligatorio — el agente pregunta por ellas |
| ¿Se detectaron sesgos estadísticos? | En Parte E con pruebas formales | En Phase 2 por intuición antes de las pruebas |

**Ejemplo concreto — Paradoja de Simpson:**

- **Enfoque clásico:** la correlación −0.235 se reporta en la matriz de correlación (Parte B/C). Sin una pregunta explícita, es fácil no notar que el signo es un artefacto de mezclar grupos.
- **Enfoque con agentes:** el agente preguntó "¿ves el mismo signo negativo dentro de cada especie?" antes de que existiera el scatter por especie. El estudiante llegó a la intuición de la Paradoja de Simpson **solo a partir del número global** — sin ver el gráfico aún.

Ese es el tipo de pensamiento que el enfoque con agentes fomenta deliberadamente.

---

### 2.3 — Selección de operaciones

| Dimensión | Clásico | Con Agentes |
|---|---|---|
| ¿Quién decide qué calcular? | El enunciado del laboratorio | El agente adaptativamente, basado en los datos |
| ¿Los gráficos son fijos o adaptativos? | Fijos (definidos en el enunciado) | Adaptativos (el runner los selecciona por varianza, correlación, cardinalidad) |
| ¿Se calculan solo las correlaciones útiles? | Todas (Pearson + Spearman por consigna) | Las que la evidencia justifica |
| ¿Hay un registro auditable de lo ejecutado? | No (el notebook es lineal pero no estructurado) | Sí: `run_log.json` con timestamps por operación |

---

### 2.4 — Reproducibilidad

| Dimensión | Clásico | Con Agentes |
|---|---|---|
| ¿Puede repetirse exactamente? | Solo si el notebook es ejecutado en orden | Sí: `RANDOM_SEED = 42`, rutas fijas, sin sobrescritura |
| ¿Los outputs tienen versión? | No | Sí: `__PHASE__TIMESTAMP` en cada artifact |
| ¿Se puede auditar qué se corrió y cuándo? | No | Sí: `run_log.json` |
| ¿Es portable a otro dataset? | Hay que reescribir el código | Sí: cambiar `--dataset` y `--origin` en la línea de comando |

---

### 2.5 — Dónde pone el tiempo el estudiante

```
Enfoque clásico:
  ~60% escribir/depurar código
  ~25% interpretar outputs
  ~15% redactar conclusiones

Enfoque con agentes:
  ~10% verificar que el runner funcione
  ~30% responder preguntas socráticas del agente
  ~35% razonar sobre hipótesis y patrones
  ~25% redactar conclusiones propias
```

El enfoque clásico invierte más tiempo en *cómo calcular*. El enfoque con agentes
invierte más tiempo en *qué significa lo calculado*.

---

## 3 — Debilidades de cada enfoque

### Debilidades del enfoque clásico

- **Riesgo de "copiar y ejecutar":** sin una estructura socrática, es posible completar el laboratorio sin haber formado ninguna hipótesis propia.
- **Análisis reactivo:** el estudiante responde lo que el enunciado pide, no necesariamente lo que los datos sugieren.
- **Sin trazabilidad:** no hay registro de qué se ejecutó, en qué orden, ni con qué parámetros.

### Debilidades del enfoque con agentes

- **Caja negra del runner:** si el estudiante no entiende lo que hace `runner.py`, puede aprobar el laboratorio sin saber qué es una correlación de Spearman o por qué se usa.
- **Dependencia del agente:** si el agente comete un error (lo que puede ocurrir), el estudiante necesita detectarlo — lo cual requiere el conocimiento que el enfoque clásico sí desarrolla.
- **Curva de setup:** la primera vez que se configura el runner y el flujo de artifacts hay fricción técnica (como los bugs encontrados en este laboratorio).

---

## 4 — Opinión del estudiante

> *"Definitivamente me siento más cómodo dando instrucciones a un agente que escribe el
> código por mí, y yo me encargo de revisar de manera general el código y centrarme en
> el análisis, pensamiento de correlaciones, hipótesis, explicar inconsistencias, etc.
> Es más esencial e importante, creo yo, aprender a analizar los datos y saber como
> tratarlos. El código también es importante pues es nuestra herramienta, pero
> considerando que con agentes se puede crear un 'observatorio de procesamiento de datos',
> el humano puede centrarse en enfocar en análisis y concluir."*

### Evaluación de esta posición

Esta posición es **metodológicamente sólida** y refleja una comprensión madura del
rol de las herramientas en el análisis de datos. Algunos matices:

**Lo que es correcto:**
- La distinción entre *herramienta* (código) y *pensamiento* (análisis) es real y
  cada vez más relevante en ciencia de datos moderna.
- La capacidad de formular hipótesis falsables, detectar sesgos y explicar inconsistencias
  es más transferible que saber escribir `sns.heatmap()`.
- El modelo "agente planifica, runner ejecuta, humano razona" es arquitectónicamente
  sano — separa responsabilidades y reduce errores.

**Lo que hay que cuidar:**
- El riesgo de la comodidad con agentes es perder la intuición de *cuándo* un resultado
  es sospechoso. Esa intuición se construye programando — aunque sea dolorosamente.
  El bug de la matriz de solo lectura de numpy, que tú lograste diagnosticar, es un
  ejemplo de por qué vale la pena entender al menos parcialmente el código.
- "Revisar de manera general el código" requiere saber suficiente para detectar errores
  no obvios. El equilibrio óptimo no es delegar el 100% del código, sino delegar el
  código *repetitivo y estructural* y mantener capacidad de auditar la lógica crítica.

**Conclusión:** el enfoque con agentes es superior para el objetivo de este laboratorio
(aprender a analizar), pero los dos enfoques son complementarios. El clásico construye
la intuición de bajo nivel; el enfoque con agentes la aplica a mayor escala.

---

## 5 — Recomendación para laboratorios futuros

| Etapa del aprendizaje | Enfoque recomendado |
|---|---|
| Primeros contactos con pandas/scipy | Clásico — escribir código propio |
| Análisis exploratorio de datasets nuevos | Con agentes — foco en hipótesis |
| Validación de modelos estadísticos | Clásico — entender cada prueba |
| Reportes reproducibles a escala | Con agentes — runner + artifacts |
| Investigación con múltiples datasets | Con agentes — portabilidad del runner |

El flujo ideal es aprender a caminar (clásico) antes de subirse al auto (agentes).
Este laboratorio hizo exactamente eso: Fase 1 clásica, Fase 2 con agentes.
