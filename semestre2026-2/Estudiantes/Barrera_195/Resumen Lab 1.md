# Análisis exploratorio del dataset Penguins

## 1. Descripción del dataset

El dataset contiene 344 filas y 7 columnas. Las variables numéricas son
`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm` y `body_mass_g`.
Las variables categóricas son `species`, `island` y `sex`.

## 2. Valores faltantes y duplicados

Se encontraron 19 valores faltantes en total. La variable `sex` contiene
11 valores faltantes. No se encontraron filas duplicadas.

## 3. Hipótesis

- `flipper_length_mm` se asocia positivamente con `body_mass_g`.
- `bill_length_mm` difiere entre las especies.
- `body_mass_g` difiere entre las especies.
- `species` e `island` están asociadas.
- `bill_depth_mm` difiere entre las especies.

## 4. Pruebas estadísticas

- Pearson para la relación entre variables numéricas.
- Kruskal-Wallis para comparar variables numéricas entre especies.
- Chi-cuadrado para evaluar la asociación entre `species` e `island`.

Los resultados deben reportarse indicando el estadístico, el p-valor y la
decisión usando un nivel de significancia de 0.05.

## 5. Conclusiones

### Hallazgos descriptivos

El dataset presenta diferencias entre especies en varias medidas corporales.
También contiene valores faltantes en las mediciones y especialmente en `sex`.

### Patrones visuales

Los histogramas muestran distribuciones multimodales. Los boxplots evidencian
diferencias entre especies y el scatterplot sugiere una asociación positiva
entre la longitud de la aleta y la masa corporal.

### Próximas hipótesis

Se propone analizar si estas diferencias se mantienen al controlar por `sex`
y estudiar la relación entre las variables numéricas dentro de cada especie.

## 6. Preguntas para el investigador

- ¿Cómo deben tratarse los valores faltantes de `sex`?
- ¿Debe `sex` utilizarse como variable de control?
- ¿Conviene realizar los análisis por separado para cada especie?
- ¿Las mediciones fueron tomadas bajo condiciones comparables?