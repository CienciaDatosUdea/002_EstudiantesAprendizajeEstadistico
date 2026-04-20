# Reporte Final: Observatorio de Datos - Laboratorio 1

**Autor:** Juan Pablo Sanchez  
**Curso:** Fisica Computacional 2  
**Fecha:** 12 de abril de 2026

## Declaracion de IA

Este reporte fue generado integramente por una Inteligencia Artificial (GitHub Copilot) mediante el analisis de artefactos de datos, bajo el protocolo de desacoplamiento Agente/Runner.

## Resumen Ejecutivo del Sistema

El sistema analizado corresponde al conjunto de pinguinos con $N=344$ observaciones y $d=7$ variables observadas: 4 numericas y 3 categoricas. Desde el marco de Physics of Data, el espacio de fases medido es mayoritariamente completo, con una entropia de observacion concentrada en la variable `sex`.

### Caracterizacion estructural

| Componente | Valor | Interpretacion |
|---|---:|---|
| Numero de observaciones | 344 | Tamano muestral adecuado para inferencia descriptiva e hipotesis basicas |
| Numero de variables | 7 | Espacio de fases mixto (continuo + discreto) |
| Variables numericas | 4 | `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` |
| Variables categoricas | 3 | `species`, `island`, `sex` |
| Duplicados | 0 | No hay redundancia exacta de estados observados |
| Baja cardinalidad | `species`, `island`, `sex` | Variables aptas para conteos, ANOVA y Chi-cuadrado |

### Faltantes y confiabilidad del vector de estado

| Variable | Faltantes | % sobre N=344 | Impacto en confiabilidad |
|---|---:|---:|---|
| `species` | 0 | 0.00% | Completitud total de etiqueta de clase |
| `island` | 0 | 0.00% | Completitud total de contexto espacial |
| `bill_length_mm` | 2 | 0.58% | Perdida puntual minima |
| `bill_depth_mm` | 2 | 0.58% | Perdida puntual minima |
| `flipper_length_mm` | 2 | 0.58% | Perdida puntual minima |
| `body_mass_g` | 2 | 0.58% | Perdida puntual minima |
| `sex` | 11 | 3.20% | Principal fuente de incertidumbre observacional |

**Lectura fisica:** la incertidumbre del sistema no es difusa, sino concentrada. El subespacio morfometrico es estable; la principal limitacion aparece en analisis condicionados por dimorfismo sexual.

## Hallazgos Estadisticos Clave

### Leyes de escala y acoplamientos

Los acoplamientos mas robustos del espacio de fases, consistentes entre Pearson y Spearman, son:

| Par de variables | Pearson | Spearman | Lectura |
|---|---:|---:|---|
| `flipper_length_mm` vs `body_mass_g` | 0.871 | 0.840 | Acoplamiento positivo dominante (ley de escala corporal) |
| `bill_length_mm` vs `flipper_length_mm` | 0.656 | 0.673 | Asociacion positiva moderada-fuerte |
| `bill_length_mm` vs `body_mass_g` | 0.595 | 0.584 | Tendencia positiva de mayor pico con mayor masa |
| `bill_depth_mm` vs `flipper_length_mm` | -0.584 | -0.523 | Acoplamiento inverso relevante |

### Patrones visuales

1. El plano `flipper_length_mm`-`body_mass_g` muestra el patron lineal mas fuerte del sistema.
2. La distribucion de `bill_length_mm` presenta bimodalidad, consistente con mezcla de estados (especies).
3. El boxplot de `bill_length_mm` por `species` sugiere separacion interespecie en la morfologia del pico.

## Tres Hipotesis Falsables

### Hipotesis 1: Interaccion aleta-masa

$$
H_0: \rho_{\text{flipper\_length\_mm},\text{body\_mass\_g}} = 0
$$

$$
H_1: \rho_{\text{flipper\_length\_mm},\text{body\_mass\_g}} > 0
$$

Prueba de validacion: correlacion de Pearson (y Spearman como robustez), con valor-p e IC95%.

### Hipotesis 2: Diferencias de longitud de pico por especie

$$
H_0: \mu_{\text{Adelie}} = \mu_{\text{Chinstrap}} = \mu_{\text{Gentoo}}
$$

$$
H_1: \exists\, i \neq j:\; \mu_i \neq \mu_j
$$

Prueba de validacion: ANOVA de un factor sobre `bill_length_mm ~ species`; post-hoc de Tukey si $p<0.05$.

### Hipotesis 3: Dependencia especie-isla

$$
H_0: \text{species} \perp \text{island}
$$

$$
H_1: \text{species} \not\perp \text{island}
$$

Prueba de validacion: Chi-cuadrado de independencia en tabla de contingencia `species x island`, con V de Cramer.

## Conclusiones Finales y Preguntas del Investigador

### Conclusiones

1. El sistema presenta lagunas de informacion pequenas pero estructuralmente relevantes, concentradas en `sex`.
2. Se observan acoplamientos lineales fuertes y estables, especialmente entre longitud de aleta y masa corporal.
3. La evidencia de multimodalidad y separacion por especie respalda un modelo de mezcla de estados en el marco Physics of Data.
4. El siguiente paso natural es pasar de descripcion a prediccion, evaluando regresion multivariada para estimar `body_mass_g`.

### Preguntas de investigacion

1. Como afecta el dimorfismo sexual (`sex`) a las leyes de escala observadas en la masa corporal?
2. Es necesaria una imputacion de faltantes antes del modelado predictivo?
3. Existen variables ambientales no registradas por isla que expliquen la segregacion de especies?

## Comparacion de Enfoques: Clasico vs. Basado en Agentes

| Criterio | Enfoque clasico (exploracion manual) | Enfoque con agentes (inferencia desacoplada) |
|---|---|---|
| Flujo de trabajo | Secuencial, guiado por decisiones ad-hoc | Separacion de roles: Runner calcula, Agente infiere |
| Reproducibilidad | Sensible al operador y al entorno | Alta trazabilidad por artefactos JSON/PNG |
| Escalabilidad | Limitada por iteracion manual | Transferible a nuevos sistemas con minima friccion |
| Riesgo de sesgo | Mayor riesgo de data snooping | Menor por restriccion de acceso a microestados |
| Interpretabilidad | Buena pero menos estandarizada | Estructura uniforme de hallazgos, hipotesis y validacion |

## Evidencia de Generalizacion

La prueba de generalizacion con el dataset `iris.csv` fue exitosa: `runner.py` identifico dinamicamente variables numericas y categoricas, ejecuto matriz de correlacion y ANOVA, y confirmo que la especie floral determina de forma significativa dimensiones fisicas. Este resultado valida la portabilidad del protocolo Agente/Runner mas alla del sistema de pinguinos.