# Hipotesis y Conclusiones Documentadas

## Hipotesis

### H1
- Enunciado: flipper_length_mm se asocia con body_mass_g.
- Pruebas sugeridas: Pearson, Spearman.
- Evidencia: correlaciones reportadas en artifacts de HYPOTHESIZE.

### H2
- Enunciado: bill_length_mm difiere entre especies.
- Pruebas sugeridas: ANOVA, Kruskal-Wallis.
- Evidencia: boxplot por especie y resumen descriptivo.

### H3
- Enunciado: species se asocia con island.
- Prueba sugerida: Chi-cuadrado de independencia.
- Evidencia: crosstab species-island.

## Conclusiones

### Capa A - Hallazgos descriptivos
- Dataset con 344 x 7.
- Faltantes bajos en total, concentrados en sex.
- No hay duplicados.

### Capa B - Patrones visuales
- Separacion por especie en scatter de variables morfometricas.
- Tendencia positiva global entre flipper_length_mm y body_mass_g.
- Diferencias visuales por especie en bill_length_mm.

### Capa C - Proximas hipotesis
- Correlaciones dentro de cada especie.
- Contraste formal species-island.
- Contraste formal bill_length_mm por especie.

## Resultado de pruebas
- Todas las pruebas ejecutadas para H1, H2 y H3 reportaron p-valor < 0.05.
- Decision: rechazar_H0 en todos los contrastes ejecutados.

## Referencia de origen
- artifacts/06_hypotheses_log__HYPOTHESIZE_AND_CONCLUDE__20260320_154100.json
- artifacts/07_conclusions__HYPOTHESIZE_AND_CONCLUDE__20260320_154100.json
- artifacts/08_tests__HYPOTHESIZE_AND_CONCLUDE__20260320_154944.json
