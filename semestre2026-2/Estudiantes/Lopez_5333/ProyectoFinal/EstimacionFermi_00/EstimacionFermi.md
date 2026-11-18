# Reto Final: Estimación de Fermi

**Curso:** Aprendizaje Estadistico 2026-2  
**Estudiantes:** Jeronimo Lopez Gomez - Jose Daniel Godoy

---

Queremos determinar ¿Cuantos mosquitos vamos a capturar durante las 4 noches en Cheté?.

Inicialmente, sin tener conocimiento de los datos históricos, podemos plantear el siguiente escenario y establecer algunas condiciones y supuestos para construir nuestra estimación:

- Se considera que la presencia de mosquitos puede ser mayor en zonas de alta humedad y temperatura, donde existen condiciones mas favorablas para su desarrollo y reproducción. Cheté se encuentra a las orillas del rio Timbiquí (2°39'32.35"N 77°37'24.61"W) aproximadamente a 20 Km de la costa pacífica por lo que consideramos relevante su entorno geográfico y climático para nuestra estimación. Por ejemplo en una zona urbana como Quibdó o directamente en la costa la cantidad de mosquitos puede verse afectada considerablemente.

- Se considera un número de puntos de colección distribuidos entre colectores ubicados a la intemperie y colectores ubicados al interior de las viviendas de la zona de estudio. Esta distribución permitirá diferenciar las capturas según el ambiente en el que se realizan.

- Suponemos que la tasa de captura de los colectores ubicados al interior de las viviendas es diferente a la de los colectores exteriores, considerando que las condiciones de temperatura, humedad y ventilación pueden influir en la presencia y actividad de los mosquitos en cada caso.

- Se consideran las horas efectivas de captura durante cada noche, teniendo en cuenta que pueden existir limitaciones operativas o físicas que impidan mantener una actividad de recolección continua durante toda la noche.

- Consideramos que, de la totalidad de mosquitos capturados en un colector durante un intervalo de tiempo definido, solo una proporción corresponde a la especie Anopheles. Por tanto, planteamos una proporción promedio de identificación de esta especie respecto al total de mosquitos capturados.

- Consideramos además que la tasa de captura puede variar durante la noche. Para nuestra estimación inicial distinguimos dos rangos horarios: desde el atardecer hasta las primeras horas de la noche, y desde las 10:00 p. m. hasta la medianoche. Esta separación nos permitirá considerar posibles diferencias en la actividad de los mosquitos a lo largo de la noche.

- Consideramos que las lluvias pueden modificar las tasas de captura según el lugar de recolección. Suponemos inicialmente que la tasa de captura de los colectores exteriores puede disminuir considerablemente en episodios de lluvia, mientras que la tasa de captura de los colectores interiores podría aumentar debido a una posible búsqueda de refugio por parte de los mosquitos. Esta relación se plantea como un supuesto.

De acuerdo a lo anterior definimos los siguientes parámetros:

- $C_i, C_e$ : Nro. de colectores ubicados en los interiores de las viviendas y en el exterior, correspondientemente.

- $T_{eff}$ : Tiempo efectivo de captura por noche, en horas.

- $N_{i,t}, N_{e,t}$ : Tasa media de captura por hora en el interior y exterior respectivamente a una franja horaria $t$

- $p_a(n)$ :  Definida como la proporción de mosquitos del género *Anopheles* por número total de mosquitos capturados $n$.

Estimamos entonces el número de mosquitos capturados en una noche $k$ como $N_{k}$, según el siguiente modelo, donde $d_{t}$ representa la duración en horas de la franja horaria $t$.

$$
N_k = \sum_{t=1}^{2} C_i \cdot N_{i,t} \cdot d_t + \sum_{t=1}^{2} C_e \cdot N_{e,t} \cdot d_t 
$$

En total para las 4 noches, la cantidad estimada de mosquitos del género *Anopheles* se puede estimar como:

$$
N_{A} = p_a \cdot \sum_{k=1}^4 N_{k}
$$
 
donde 

$$
T_{eff} = d_1 + d_2, \ \  t=1,2
$$


Dado lo anterior establezcamos el siguiente caso: 

$C_i \rightarrow$ 2 colectores

$C_e \rightarrow$ 3 colectores

$d_1 \rightarrow$	3 horas (Últimas horas del atardecer y primeras horas de la noche)

$d_2 \rightarrow$	2 horas	(Desde las 10:00 p.m. hacia las 11:59 p.m.)

$p_a \rightarrow$	0.65 (Supuesto: El 65% de las capturas corresponden a Anopheles)

Consideremos además las siguientes tasas de mosquitos capturados

$(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (6, 4, 7, 3) \frac{mosquitos}{hora}$

De acuerdo a este escenario se obtienen los siguientes resultados:

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 52 | 208 |
| Capturas exteriores | 81 | 324 |
| **Total de mosquitos** | **133** | **532** |
| Proporción de *Anopheles* | 65 % | 65 % |
| **Anopheles estimados** | **86.45** | **346** |


Establezcamos a continuación un rango, considerando casos de minima recolección y máxima recolección

para el caso donde la recolección de mosquitos es muy pobre podemos suponer los siguientes valores, variando principalmente la tasa de recolección y la duración de las franjas horarias:

$C_i \rightarrow$ 2 colectores

$C_e \rightarrow$ 3 colectores

$d_1 \rightarrow$	2 horas (Últimas horas del atardecer y primeras horas de la noche)

$d_2 \rightarrow$	1 horas	(Desde las 10:00 p.m. hacia las 11:59 p.m.)

$p_a \rightarrow$	0.65 (Supuesto: El 65% de las capturas corresponden a Anopheles)

Consideremos además las siguientes tasas de mosquitos capturados

$(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (4, 3, 5, 3) \frac{mosquitos}{hora}$

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 22 | 88 |
| Capturas exteriores | 39 | 156 |
| **Total de mosquitos** | **61** | **244** |
| Proporción de *Anopheles* | 65 % | 65 % |
| **Anopheles estimados** | **39** | **158** |

Posteriormente en el caso de alta recolección

$C_i \rightarrow$ 2 colectores

$C_e \rightarrow$ 3 colectores

$d_1 \rightarrow$	3 horas (Últimas horas del atardecer y primeras horas de la noche)

$d_2 \rightarrow$	3 horas	(Desde las 10:00 p.m. hacia las 11:59 p.m.)

$p_a \rightarrow$	0.65 (Supuesto: El 65% de las capturas corresponden a Anopheles)

Consideremos además las siguientes tasas de mosquitos capturados

$(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (8, 7, 7, 4) \frac{mosquitos}{hora}$

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 90 | 360 |
| Capturas exteriores | 99 | 396 |
| **Total de mosquitos** | **189** | **756** |
| Proporción de *Anopheles* | 65 % | 65 % |
| **Anopheles estimados** | **122** | **490** |

Obtengamos finalmente el rango estimado de mosquitos a obtener en la expedición de acuerdo a los supuestos y condiciones anteriormente establecidas.


| Escenario | Mosquitos por noche | Mosquitos en 4 noches | *Anopheles* estimados |
|---|---:|---:|---:|
| Mínima recolección | 61 | 244 | 158 |
| Caso central | 133 | 532 | 345 |
| Máxima recolección | 189 | 756 | 490 |


