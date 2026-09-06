# Registro de reflexión

## Ciclo 1
**Hipótesis:** En la muestra completa, bill_length_mm y bill_depth_mm presentan una asociación lineal.
**Análisis ejecutado:** Se eliminaron filas con bill_length_mm o bill_depth_mm faltantes y se calculó Pearson sobre toda la muestra válida.
**Resultado numérico:** r=-0.235, p=1.12e-05, n=342
**Objeciones:**
- Se ignoró la variable de agrupación species; la relación debe revisarse por especie.
- La muestra completa puede ser heterogénea y mezclar especies puede ocultar diferencias.
**Bloqueante:** true
**Qué cambió:** Análisis inicial de la muestra completa.
**Conclusión:** En la muestra completa, la correlación de Pearson entre bill_length_mm y bill_depth_mm fue r=-0.235, p=1.12e-05 y n=342. Esto describe una asociación lineal del conjunto total y no demuestra causalidad.

## Ciclo 2
**Hipótesis:** La asociación entre las medidas puede cambiar entre especies.
**Análisis ejecutado:** Se usaron las filas numéricas válidas, se separaron por especie para reducir la heterogeneidad y se calculó Pearson dentro de cada grupo.
**Resultado numérico:** Adelie: r=0.391, p=6.67e-07, n=151; Chinstrap: r=0.654, p=1.53e-09, n=68; Gentoo: r=0.643, p=1.02e-15, n=123
**Objeciones:**
- Ninguna.
**Bloqueante:** false
**Qué cambió:** Se repitió Pearson por especie y se evitó una afirmación causal.
**Conclusión:** Por especie, Pearson entre bill_length_mm y bill_depth_mm dio Adelie: r=0.391, p=6.67e-07, n=151; Chinstrap: r=0.654, p=1.53e-09, n=68; Gentoo: r=0.643, p=1.02e-15, n=123. Estos resultados describen asociaciones lineales dentro de cada especie y no demuestran causalidad.
