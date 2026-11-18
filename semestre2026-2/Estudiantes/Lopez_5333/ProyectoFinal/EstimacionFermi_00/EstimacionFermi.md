# Reto Final: Estimación de Fermi

**Curso:** Aprendizaje Estadistico 2026-2  
**Estudiantes:** Jeronimo Lopez Gomez - Jose Daniel Godoy

---

Queremos determinar ¿Cuantos mosquitos vamos a capturar durante las 4 noches en Cheté?.

Inicialmente, sin tener conocimiento de los datos históricos, podemos plantear el siguiente escenario y establecer algunas condiciones y supuestos para construir nuestra estimación:

- Se considera que la presencia de mosquitos puede ser mayor en zonas de alta humedad y temperatura, donde existen condiciones mas favorablas para su desarrollo y reproducción. Cheté se encuentra a las orillas del rio Timbiquí (2°39'32.35"N 77°37'24.61"W) aproximadamente a 20 Km de la costa pacífica, en plena temporada de lluvias, por lo que consideramos relevante su entorno geográfico y climático para nuestra estimación. Por ejemplo en una zona urbana como Quibdó o directamente en la costa la cantidad de mosquitos puede verse afectada considerablemente.

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

- $p_a$ :  Definida como la proporción de mosquitos del género *Anopheles* por número total de mosquitos capturados.

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

- $C_i = 2$ colectores interiores, $C_e = 3$ colectores exteriores (5 puntos en total).
- $d_1 = 3\text{ horas}$ (aprox. 6:00 p. m. – 9:00 p. m.), $d_2 = 2\text{ horas}$ (aprox. 9:00 p. m. – 11:00 p. m.), para $T_{eff} = 5\text{ horas/noche}$.
- $p_a = 0.65$ (65 % de las capturas corresponden a *Anopheles*).
- Tasas de captura: $(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (6, 4, 7, 3)\ \frac{\text{mosquitos}}{\text{hora}}$.

Cálculo por noche:
- Interior: $2 \cdot (6 \cdot 3 + 4 \cdot 2) = 2 \cdot 26 = 52$
- Exterior: $3 \cdot (7 \cdot 3 + 3 \cdot 2) = 3 \cdot 27 = 81$

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 52 | 208 |
| Capturas exteriores | 81 | 324 |
| **Total de mosquitos** | **133** | **532** |
| Proporción de *Anopheles* ($p_a$) | 65 % | 65 % |
| ***Anopheles* estimados** | **86** *(86.45)* | **346** *(345.8)* |


Establezcamos a continuación un rango, considerando casos de minima recolección y máxima recolección

Suponemos menor actividad de mosquitos, interrupciones operativas/climáticas que reducen las horas efectivas a $T_{eff} = 3\text{ horas/noche}$ y una menor proporción de *Anopheles* frente a otros mosquitos:

- $C_i = 2$ colectores, $C_e = 3$ colectores.
- $d_1 = 2\text{ horas}$, $d_2 = 1\text{ hora}$ ($T_{eff} = 3\text{ horas/noche}$).
- $p_a = 0.50$.
- Tasas de captura: $(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (4, 3, 5, 3)\ \frac{\text{mosquitos}}{\text{hora}}$.

Cálculo por noche:
- Interior: $2 \cdot (4 \cdot 2 + 3 \cdot 1) = 2 \cdot 11 = 22$
- Exterior: $3 \cdot (5 \cdot 2 + 3 \cdot 1) = 3 \cdot 13 = 39$

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 22 | 88 |
| Capturas exteriores | 39 | 156 |
| **Total de mosquitos** | **61** | **244** |
| Proporción de *Anopheles* ($p_a$) | 50 % | 50 % |
| ***Anopheles* estimados** | **31** *(30.5)* | **122** *(122.0)* |

*(Nota: si $p_a$ se mantuviera en 65 %, serían 40 *Anopheles* por noche y 159 en las 4 noches).*

Suponemos alta densidad poblacional, jornadas completas hasta medianoche ($T_{eff} = 6\text{ horas/noche}$), mayor captura intradomiciliaria por búsqueda de refugio ante lluvia ligera y alta dominancia de *Anopheles*:

- $C_i = 2$ colectores, $C_e = 3$ colectores.
- $d_1 = 3\text{ horas}$, $d_2 = 3\text{ horas}$ ($T_{eff} = 6\text{ horas/noche}$).
- $p_a = 0.80$.
- Tasas de captura: $(N_{i,1}, N_{i,2}, N_{e,1}, N_{e,2}) = (8, 7, 7, 4)\ \frac{\text{mosquitos}}{\text{hora}}$.

Cálculo por noche:
- Interior: $2 \cdot (8 \cdot 3 + 7 \cdot 3) = 2 \cdot 45 = 90$
- Exterior: $3 \cdot (7 \cdot 3 + 4 \cdot 3) = 3 \cdot 33 = 99$

| Descripción | Por noche | Cuatro noches |
|---|---:|---:|
| Capturas interiores | 90 | 360 |
| Capturas exteriores | 99 | 396 |
| **Total de mosquitos** | **189** | **756** |
| Proporción de *Anopheles* ($p_a$) | 80 % | 80 % |
| ***Anopheles* estimados** | **151** *(151.2)* | **605** *(604.8)* |

*(Nota: si $p_a$ se mantuviera en 65 %, serían 123 *Anopheles* por noche y 491 en las 4 noches).*

Obtengamos finalmente el rango estimado de mosquitos a obtener en la expedición de acuerdo a los supuestos y condiciones anteriormente establecidas.


| Escenario | Mosquitos totales / noche | Mosquitos totales (4 noches) | *Anopheles* estimados (4 noches) |
|---|---:|---:|---:|
| Mínima recolección | 61 | 244 | 122 |
| **Caso central** | **133** | **532** | **346** |
| Máxima recolección | 189 | 756 | 605 |



Finalmente el estimado de mosquitos a recolectar durante las 4 noches sin tener el contexto de las recolecciones anteriores, se ubica en **532 mosquitos** y de estos, la idea es que alrededor de estos **346 sean de la especie Anopheles**.

Respecto a la sensibilidad de lo anteriormente afirmado, que consideramos como el caso central, analizemos a continuación de forma aislada como afecta la variación de los 4 principales parámetros la cantidad de mosquitos recolectados, manteniendo los parámetros del caso central evaluado de manera fija.


| Supuesto evaluado (variando solo ese factor) | Rango evaluado | *Anopheles* en 4 noches |
|---|---|---:|
| **Horas efectivas por noche ($T_{eff}$)** | **3 Horas** ($d_1,d_2 = 2,2$) a **10 Horas** ($d_1, d_2 = 5,5$) | 260 – 650 |
| **Número de colectores ($C_i, C_e$)** | $(1, 2)$ a $(3, 4)$ colectores | 208 – 484 | 
| **Tasas de captura ($N_{i,t}, N_{e,t}$)** | Mínimas $(4,3,5,3)$ a máximas $(8,7,7,4)$ | 257 – 424 |
| **Proporción de *Anopheles* ($p_a$)** | 50 % a 80 % | 266 – 426 |


Nótese entonces que el supuesto que mas modifica la cantidad de *Anopheles* recolectados es el tiempo total efectivo utilizado para la recolección marcando una diferencia sustancial de **390** mosquitos, en la misma linea se registra que el número de colectores también tiene un efecto importante con una diferencia de **276** *Anopheles* de diferencia, por lo que estos 2 primeros parámetros serán los mas críticos en la estimación inicial.

