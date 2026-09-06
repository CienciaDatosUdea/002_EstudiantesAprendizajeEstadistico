# Registro de interacción con el agente

## Resumen general

Este documento registra la interacción entre el agente y el runner a lo largo de las tres fases del laboratorio:

1. OBSERVE
2. DESCRIBE
3. HYPOTHESIZE_AND_CONCLUDE

La lógica del flujo fue:
- el agente propone qué analizar,
- el runner ejecuta la operación correspondiente,
- se generan artifacts JSON/PNG,
- el agente interpreta solo esos resultados y redacta conclusiones.

---

## Fase 1: OBSERVE

### Qué le pregunté al agente

Se solicitó que propusiera qué operaciones de observación inicial debía ejecutar el runner sobre el dataset, sin asumir nada previo.

La instrucción del laboratorio pedía revisar, como mínimo:
- forma del dataset,
- tipos de variable,
- nulos por columna,
- duplicados,
- cardinalidad de cada variable.

### Qué propuso el agente

El agente recomendó validar la estructura del dataset antes de inferir patrones. En concreto propuso:
- revisar filas y columnas,
- comprobar nombres de columnas y tipos inferidos,
- detectar nulos por columna y proporciones,
- identificar filas duplicadas,
- calcular cardinalidad por columna,
- detectar valores inconsistentes o mezclados en tipo.

### Qué ejecuté en el runner

Se ejecutó la función:

```python
correr_observe(df)
```

La salida fue el artifact:
- `00_raw_profile.json`

### Resultado generado

El artifact generado reportó:
- `n_filas`: 344
- `n_columnas`: 7
- columnas: `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex`
- nulos por columna:
  - `species`: 0
  - `island`: 0
  - `bill_length_mm`: 2
  - `bill_depth_mm`: 2
  - `flipper_length_mm`: 2
  - `body_mass_g`: 2
  - `sex`: 11
- duplicados: 0
- cardinalidad:
  - `species`: 3
  - `island`: 3
  - `sex`: 2

### Decisión tomada

La fase OBSERVE quedó cerrada con la validación de estructura y calidad básica del dataset, sin asumir hipótesis de negocio ni relaciones entre variables.

---

## Fase 2: DESCRIBE

### Qué le pregunté al agente

Se solicitó que propusiera qué estadísticos descriptivos, tablas cruzadas, correlaciones y gráficos explorar a continuación, sin sesgo predefinido y con base en la información de `00_raw_profile.json`.

### Qué propuso el agente

El agente recomendó:

#### Estadísticos descriptivos
- resumen numérico para variables continuas:
  - conteo,
  - media,
  - mediana,
  - desviación estándar,
  - mínimo,
  - máximo,
  - cuartiles,
  - missing por variable.
- resumen de frecuencias para variables categóricas:
  - conteos,
  - porcentajes,
  - nulos.

#### Tablas cruzadas
- `species x island`
- `species x sex`
- `island x sex`

#### Correlaciones
- matriz de correlación Pearson entre:
  - `bill_length_mm`
  - `bill_depth_mm`
  - `flipper_length_mm`
  - `body_mass_g`
- también se contempló Spearman como complemento.

#### Gráficos sugeridos
- barras de variables categóricas,
- histogramas de numericas,
- boxplot de `body_mass_g` por `species`,
- boxplot de `bill_length_mm` por `island`,
- scatter de `bill_length_mm` vs `flipper_length_mm` coloreado por `species`,
- heatmap de correlación.

### Qué ejecuté en el runner

Se ejecutó la función:

```python
correr_describe(
    df,
    ["species", "island", "sex"],
    ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"],
    [("species", "island"), ("species", "sex"), ("island", "sex")],
)
```

### Resultados generados

Los artifacts producidos fueron:
- `04_descriptive_stats.json`
- `05_visual_registry.json`

Además, se generaron los PNG en `artifacts/`:
- `05_species_bar.png`
- `05_island_bar.png`
- `05_sex_bar.png`
- `05_bill_length_mm_hist.png`
- `05_bill_depth_mm_hist.png`
- `05_flipper_length_mm_hist.png`
- `05_body_mass_g_hist.png`
- `05_body_mass_g_by_species_boxplot.png`
- `05_bill_length_mm_by_island_boxplot.png`
- `05_bill_length_mm_vs_flipper_length_mm_by_species_scatter.png`
- `05_numeric_correlation_heatmap.png`

### Decisión tomada

La fase DESCRIBE quedó respaldada por un conjunto de descriptivos y gráficos que permitieron observar patrones preliminares sin imponer hipótesis previas.

---

## Fase 3: HYPOTHESIZE_AND_CONCLUDE

### Qué le pregunté al agente

Se pedía que, con la evidencia de los artifacts descriptivos y visuales como única base, propusiera:
- al menos 3 hipótesis falsables,
- la prueba estadística adecuada para cada una,
- conclusiones en tres capas:
  1. hallazgos descriptivos,
  2. patrones visuales,
  3. próximas hipótesis a probar,
- preguntas para un investigador humano.

### Qué propuso el agente

El agente propuso estas hipótesis:

#### Hipótesis 1
- `species` y `body_mass_g` están asociadas.
- Prueba recomendada: ANOVA de una vía.
- Evidencia real posterior: F = 343.626, p = 2.89e-82.

#### Hipótesis 2
- `species` y `island` están asociadas.
- Prueba recomendada: chi-cuadrado de independencia.
- Evidencia real posterior: chi2 = 299.55, dof = 4, p = 1.35e-63.

#### Hipótesis 3
- `bill_length_mm` y `flipper_length_mm` están asociadas.
- Prueba recomendada: correlación de Pearson.
- Evidencia real posterior: r = 0.656, p = 1.74e-43.

### Qué ejecuté en el runner

Se ejecutó esta sección en el bloque principal:

```python
resultados = [
    prueba_anova(df, "species", "body_mass_g"),
    prueba_chi2(df, "species", "island"),
    prueba_pearson(df, "bill_length_mm", "flipper_length_mm"),
]
correr_tests(resultados)
```

### Resultado generado

El artifact generado fue:
- `08_tests.json`

Con el siguiente contenido real:

```json
[
  {
    "tipo": "anova",
    "col_grupo": "species",
    "col_valor": "body_mass_g",
    "f": 343.626,
    "p_valor": 2.8923681333773435e-82
  },
  {
    "tipo": "chi2",
    "col1": "species",
    "col2": "island",
    "chi2": 299.55,
    "p_valor": 1.354573829719252e-63,
    "dof": 4
  },
  {
    "tipo": "pearson",
    "col1": "bill_length_mm",
    "col2": "flipper_length_mm",
    "r": 0.656,
    "p_valor": 1.7439736176205688e-43
  }
]
```

### Conclusión final del agente

La evidencia apoyó las tres hipótesis planteadas:
- `species` y `body_mass_g`: sí hay evidencia estadística de diferencia.
- `species` e `island`: sí hay asociación.
- `bill_length_mm` y `flipper_length_mm`: sí hay asociación lineal positiva.

Sin embargo, el laboratorio exige no afirmar causalidad. Por eso, la conclusión válida es:
- hay relación estadística detectable,
- pero no se puede afirmar que una variable cause directamente a la otra.

### Preguntas para un investigador humano

- ¿La asociación entre especie e isla refleja distribución ecológica real o un sesgo de muestreo?
- ¿La diferencia en masa corporal puede explicarse por características biológicas de la especie o por variables no observadas?
- ¿La correlación entre bill_length_mm y flipper_length_mm es consistente entre especies o varía por grupo?
- ¿La variable sex debe considerarse como posible factor de confusión?
- ¿Conviene controlar por isla o por especie al comparar morfologías? 

---

## Cierre del registro

El ciclo completo quedó ejecutado con la siguiente secuencia de artifacts:

1. `00_raw_profile.json` — perfil inicial del dataset
2. `04_descriptive_stats.json` — estadísticas descriptivas y tablas cruzadas
3. `05_visual_registry.json` — nombres de gráficas y archivos PNG
4. `08_tests.json` — pruebas estadísticas para las hipótesis principales

Esto deja documentado el flujo completo de interacción entre el agente y el runner, cumpliendo con la estructura de trabajo del laboratorio.
