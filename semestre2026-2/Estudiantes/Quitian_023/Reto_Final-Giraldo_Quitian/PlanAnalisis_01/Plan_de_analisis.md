# Plan de análisis

Felipe Quitian Gallego, Miguel Angel Giraldo

## De vuelta con los datos

Seguimos el plan paso a paso en el notebook (SintesisFinal_02/Reto_Final.ipynb). En cada paso respondemos las preguntas que nos habíamos hecho, decimos qué decidimos y, cuando algo cambió respecto al plan, lo decimos.

### 1. Las columnas que nos dan información

El archivo trae 1502 filas y 69 columnas, con 229394 mosquitos (194356 An. neivai, 34424 An. albimanus y 614 An. apicimacula), las mismas cifras del resumen del estudio. Cada fila no es un mosquito ni una noche, es la suma de una especie en una campaña de un conglomerado. De las 69 columnas, 31 tienen un solo valor (por ejemplo, todas las filas son hembras y el hábitat siempre es En domicilio). Las que cambian y usamos son el conglomerado y la visita (eventID), las fechas, el esfuerzo en noches y horas (samplingEffort), la especie, el conteo, la localidad y el municipio, las coordenadas y dos columnas que no habíamos pensado anotar en nuestra hoja de campo: quién registra (recordedBy) y quién identifica la especie (identifiedBy).

Antes de usarlas revisamos las filas crudas. El problema que más importó fue un error de año. Seis campañas de Timbiquí (BrazoC 19 a 21 y Coteje1 20 a 22) dicen 2022 en eventDate y enero a marzo de 2023 en verbatimEventDate. Sin corregirlo, campañas de 2023 quedaban como si fueran de 2022. Corregimos el año usando el orden de las visitas de cada conglomerado. Además encontramos 20 filas sin eventDate, identificadores con guion bajo que creaban 15 conglomerados falsos (de 55 nombres quedan 40), 15 campañas en las que cada especie trae su propio número de noches, 7 pares de filas repetidas y una misma coordenada repetida en tres conglomerados distintos. En cada caso dejamos escrita la decisión en el notebook.

### 2. ¿Todos los datos, Timbiquí o solo Cheté?

¿Cuántas expediciones se han realizado? Definimos una expedición, o campaña, como la visita de dos recolectores a un conglomerado durante varias noches. En todo el estudio hay 879 campañas en 40 conglomerados de tres municipios. Timbiquí tiene 349 campañas en 16 conglomerados, y cada punto de Cheté fue visitado 22 veces entre enero de 2021 y marzo de 2023, casi siempre una vez al mes.

¿Se realizaron en las mismas fechas? Sí. Los tres municipios comparten 19 rondas mensuales (Guapi y Santa Bárbara tienen una más), así que sus diferencias no vienen de haberlos muestreado en meses distintos.

¿Cómo cambia la cantidad de mosquitos según la población? Los tres municipios son dominados por An. neivai, pero An. albimanus, la especie de Cheté, es el 22% de lo capturado en Timbiquí, el 20% en Santa Bárbara y apenas el 4% en Guapi. La tasa mediana va de 34 mosquitos por noche en Guapi a 72 en Timbiquí y 91 en Santa Bárbara.

¿Qué variable pesa más y pesan lo mismo en las diferentes regiones? Para medirlo usamos el R²: reemplazamos cada campaña por el promedio de su grupo (su conglomerado o su ronda) y medimos qué fracción de la variación desaparece,

R² = 1 − Σ(yᵢ − ȳ_grupo)² / Σ(yᵢ − ȳ)².

No pesan igual. En Guapi y Timbiquí el conglomerado explica más del 40% de la variación y la ronda poco, es decir, importa dónde se captura. En Santa Bárbara pasa al revés, aunque con solo 4 conglomerados.

| Municipio | Conglomerados | Campañas antes de 2023 | Tasa mediana por noche | R² del conglomerado | R² de la ronda |
|---------------|-------------------|----------------------------|----------------------------|-------------------------|--------------------|
| Guapi         | 20                | 382                        | 34                         | 0.42                    | 0.11               |
| Santa Bárbara | 4                 | 76                         | 91                         | 0.15                    | 0.38               |
| Timbiquí      | 16                | 301                        | 72                         | 0.44                    | 0.13               |

Con esto decidimos trabajar con Timbiquí. Todo el estudio mezcla municipios donde las variables no pesan igual, Guapi casi no tiene la especie de Cheté y solo Cheté son dos puntos, muy pocos para aprender una regla. Timbiquí es el municipio de Cheté y tiene 16 conglomerados, 4 de ellos de An. albimanus. Cambio respecto al plan: estas comparaciones las hicimos solo con las campañas anteriores a 2023, porque movimos la separación de los datos de validación antes de este paso (ver el paso 4).

### 3. ¿Está completo el dataset y los días son siempre los mismos?

¿Hay expediciones donde no se tomaron datos? En Timbiquí faltan 3 campañas antes de 2023, 2 en septiembre de 2021 y 1 en diciembre de 2021, y ninguna campaña tiene cero mosquitos (la más pequeña tiene 1). Como un archivo de ocurrencias solo guarda presencias, algunas de las que faltan podrían ser ceros que no quedaron registrados. No se puede saber, así que las tratamos como desconocidas.

¿Los días de la expedición siempre son los mismos? No. Las campañas van de 1 a 6 noches (208 de 301 son de 3 o 4), las noches de una campaña no son seguidas (en Cheté una campaña de 4 noches se reparte en unos 17 días) y la línea base, en enero y febrero de 2021, tuvo noches de 14 horas en vez de 12. Por eso el criterio de normalización es dividir por el esfuerzo:

esfuerzo = noches × horas / 12, tasa = total / esfuerzo, y = log10(tasa + 1).

La tasa son mosquitos por noche estándar de 12 horas en un punto con dos recolectores. Trabajamos con el logaritmo porque el ruido es multiplicativo. En cada conglomerado la desviación estándar del total crece en proporción a su media (pendiente 1.03 en escala logarítmica), así que en escala lineal los sitios grandes dominarían cualquier promedio. En logaritmo un cambio del 50% pesa lo mismo en cualquier sitio, y una diferencia de 0.3 es un factor 2.

### 4. La separación de los datos

Separamos las 48 campañas de Timbiquí desde el 1 de enero de 2023 (enero a marzo, los 16 conglomerados). No entraron en la exploración ni en la construcción del modelo, y se usaron una sola vez al final. Quedan 301 campañas de 2021 y 2022 para trabajar.

Esto cambió en dos cosas respecto al plan. Primero, hicimos la separación antes de comparar las tres opciones de datos y no después, para que esa elección no mirara 2023. Segundo, en el plan llamábamos validación a las campañas de 2023. Al construir el modelo vimos que hacían falta tres conjuntos: 2021 para entrenar, 2022 para validar, prediciendo cada ronda con las anteriores, y 2023 como prueba final. La razón es que las campañas no son independientes: una campaña se parece a la anterior del mismo conglomerado (correlación 0.48), así que repartirlas al azar haría que el modelo aprendiera del futuro.

### 5. ¿Cómo se segmentan los datos?

¿Importa más la ubicación? Sí, es lo que más importa. Dentro de Timbiquí los conglomerados se separan por especie en 4 sitios de An. albimanus (Chete 1, Chete 2, Coteje 1 y Coteje 2, juntos al sureste), 2 mixtos y 10 de An. neivai. Cheté no representa al municipio, es uno de los pocos lugares donde casi todo es An. albimanus.

![](figuras/fig_nb_1.png)

*Figura 4. Conglomerados de Timbiquí según su especie dominante. El tamaño crece con la tasa típica.*

Graficamos las distribuciones de y según cada atributo de forma simultánea y medimos cada uno con su R², comparado con el que sale al barajar las etiquetas al azar (una prueba de permutaciones). El conglomerado explica el 44% de la variación (5% por azar), la localidad el 23%, la especie dominante el 16%, la ronda (la fecha de la expedición) el 13% y el mes el 7%. El número de noches, la fase del estudio y el período de las personas no se distinguen del azar. Que el número de noches no importe es lo que esperábamos si dividir por el esfuerzo funciona.

![](figuras/fig_nb_2.png)

*Figura 5. Cuánto de la variación de la tasa explica cada atributo en Timbiquí, comparado con lo que explica por azar.*

¿Importa la persona que hace la captura? No lo pudimos separar. recordedBy siempre incluye a la misma recolectora, a la que se suma un segundo nombre desde 2022, e identifiedBy cambia una sola vez, en octubre de 2021. Son tres períodos seguidos que no se mezclan, así que cualquier diferencia entre personas se confunde con la estación y con la fase del estudio. Lo que sí revisamos es si al cambiar quien identifica saltó la proporción de An. neivai, y no: el 73% de los otros cortes posibles da un salto igual o mayor.

### 6. Correlación entre variables y PCA

Describimos cada conglomerado con su nivel, su variabilidad entre visitas, su tendencia, su fracción de An. neivai y su ubicación. La matriz de correlación muestra un gradiente geográfico: los sitios del norte y del oeste son de An. neivai y los del sureste, Cheté y Coteje, de An. albimanus (correlaciones de 0.76 y −0.65). El PCA resume ese gradiente en su primera componente, con el 48% de la varianza.

¿Es PCA una alternativa interesante? Para describir sí, para el modelo no. Con 16 conglomerados no hay suficientes sitios para reemplazar atributos por componentes, y lo que resume la primera componente ya entra al modelo por el historial de cada conglomerado. Lo que nos deja es que Cheté está en un extremo del municipio, y eso va a pesar en el modelo.

### 7. Cheté y la estimación de Fermi

Chete 1 captura una mediana de 72 mosquitos por noche y Chete 2 de 19, los dos casi solo An. albimanus (98.5% y 96.2%). No detectamos una tendencia en el tiempo. En el 61% de las campañas pasa de 1 mosquito por recolector y hora, así que no hay un tope de uno por hora: se captura lo que llega, y el frasco por hora es la forma de empacar.

![](figuras/fig_nb_3.png)

*Figura 6. Mosquitos por noche en los dos puntos de Cheté. La franja gris es la línea base.*

Volviendo a nuestra Fermi, el esfuerzo que supusimos coincide con el del estudio en horas nominales: 8 trampas por 6 horas son 48 horas de captura por noche, y 2 puntos por 2 recolectores por 12 horas también. Llevando cada ronda a 4 noches de 12 horas, los dos puntos juntos suman una mediana de 417 mosquitos (entre 19 y 896). La diferencia con nuestros 192 viene de la tasa, que es de 2.2 mosquitos por recolector y hora y no 1. La Fermi revisada para Cheté es

N = 2 puntos × 2 recolectores × 12 h × 4 noches × 2.2 ≈ 420 mosquitos.

Nuestro rango de 96 a 576 contuvo 14 de las 19 rondas. La tasa sí era el supuesto que más pesaba, pero no es un número de la región sino de cada punto: Chete 1 tiene una tasa típica de 3 por recolector y hora y Chete 2 de 0.8, y entre los dos cubren casi todo el rango que pusimos por desconocimiento. La cercanía al agua, que creíamos lo más importante, no se puede comprobar con estos datos.

### 8. El hallazgo que pusimos a prueba

Chete 1 captura unas 3.8 veces más que Chete 2 (intervalo del 95% de 2.7 a 5.4) en 18 de 19 rondas, a 227 metros de distancia. Lo probamos barajando las etiquetas de los dos puntos dentro de cada ronda (p \< 0.0001), y sigue en pie con cuatro cambios de análisis. Pero en la línea base la razón era 1.1, y la diferencia apareció después. Como no sabemos qué intervención recibió cada punto, no podemos separar el sitio de la intervención. La consecuencia práctica es que los dos puntos no se pueden promediar, y el modelo los predice por separado.

### 9. El modelo para la próxima expedición

¿Qué predecimos? El total de mosquitos de la próxima campaña de cada punto, sabiendo de antemano cuántas noches dura. De esa cifra dependen la sílica gel y las horas de laboratorio. Los frascos no los predecimos, porque los fija el protocolo: de la descripción inferimos un frasco por recolector y por hora, 96 por conglomerado, pero hay que confirmarlo con el equipo. Una predicción es buena si lo observado queda entre la mitad y el doble de lo predicho, así que medimos la fracción de campañas dentro de un factor 2. Para cada modelo damos también su error relativo, la suma de \|observado − predicho\| dividida por la suma de lo observado: de cada 100 mosquitos capturados, en cuántos se equivocó.

¿Con qué variables? Solo con lo que se sabe antes de la visita, y guiados por la segmentación: como pesa el sitio, el promedio histórico del conglomerado y su campaña anterior; como pesa la especie, su fracción histórica de An. neivai; y como pesan un poco la ronda y el mes, una señal regional y el mes. Para no quedarnos con todas sin razón hicimos una selección hacia adelante en la validación de 2022, con una regla escrita antes de ver la tabla: el conjunto más pequeño a menos de 0.02 del mejor. Quedaron dos variables, el promedio histórico del punto y su última campaña. El mes y la señal regional no aportaron lo suficiente.

¿Qué modelos consideramos? En orden de complejidad, tres referencias sencillas (la tasa típica de Timbiquí, repetir la última campaña y el promedio del sitio), una regresión lineal con la ecuación normal y una regresión de Poisson,

μ = esfuerzo · exp(θ₀ + θ₁ · histórico + θ₂ · última campaña),

entrenada con descenso del gradiente. Elegimos Poisson porque los datos son conteos con efectos multiplicativos. Así como mínimos cuadrados es la máxima verosimilitud con ruido gaussiano, la verosimilitud de Poisson da una función de coste cuyo gradiente, (μ − y)x, tiene la misma forma que el de la regresión logística. Además nunca predice tasas negativas, cosa que la regresión lineal sí hizo en 9 campañas. Los conteos varían 141 veces más de lo que supone la Poisson, y eso no daña la predicción del promedio pero sí su intervalo, así que el intervalo lo construimos con los errores del modelo en la validación.

| Modelo, validación 2022 (160 campañas) | Dentro de un factor 2 | Error típico | Error relativo |
|--------------------------------------------|---------------------------|------------------|--------------------|
| Tasa típica de Timbiquí                    | 0.49                      | factor 2.05      | 63%                |
| Repetir la última campaña                  | 0.62                      | factor 1.67      | 52%                |
| Promedio del sitio                         | 0.66                      | factor 1.64      | 52%                |
| Regresión lineal                           | 0.61                      | factor 1.65      | 50%                |
| Regresión de Poisson                       | 0.70                      | factor 1.64      | 48%                |

¿Qué dice el modelo? Como las dos variables son logaritmos de tasas, el modelo sin estandarizar se puede escribir como

tasa predicha ≈ 3.93 · H<sup>0.50</sup> · U<sup>0.24</sup>,

donde H es la media geométrica de (tasa + 1) en las visitas anteriores del punto y U es (tasa + 1) en su última visita. Los exponentes suman 0.74, menos de 1, así que un sitio que captura 10 veces más que otro recibe una predicción solo unas 5.5 veces mayor (10<sup>0.74</sup>). Por eso el modelo acerca cada sitio al promedio de Timbiquí: se queda corto en los sitios altos y se pasa en los bajos, y en Cheté se queda corto en Chete 1 y se pasa en Chete 2. Le gana a la tasa típica y a la regresión lineal, pero no se distingue del promedio del sitio.

¿Está aprendiendo o memorizando? No hay sobreajuste. El acierto en entrenamiento (62%) no supera al de validación (70%) ni al de prueba (79%), y la curva de aprendizaje casi no mejora al agregar rondas. El modelo está limitado por el ruido entre campañas, no por la cantidad de datos ni por su complejidad, así que un modelo más flexible difícilmente lo mejoraría con estas columnas.

![](figuras/fig_nb_4.png)

*Figura 7. Curva de aprendizaje: acierto en entrenamiento y en la validación de 2022 al entrenar con cada vez más rondas de 2021.*

¿Cómo le fue en 2023? En las 48 campañas reservadas el promedio del sitio quedó algo mejor que el modelo (81% contra 79% dentro del factor 2) y el intervalo del 80% cubrió al 88%. En error relativo el promedio del sitio se equivocó en 39 de cada 100 mosquitos, la Poisson en 40, la lineal en 42, la persistencia en 48 y la tasa típica en 60. En Cheté el modelo predijo más de lo observado en 5 de las 6 campañas.

¿Cuántos mosquitos traeremos? Para la próxima visita de 4 noches el modelo predice 284 mosquitos en Chete 1 y 111 en Chete 2, unos 406 en total, con un 80% de probabilidad de estar entre 196 y 825 y un 4% de superar 1000. Por el sesgo, la cifra de Chete 2 probablemente es alta, y el promedio del sitio (322 para los dos puntos) es una cota baja razonable. Comparada con nuestra Fermi de 192, la cifra es el doble, por la tasa.

![](figuras/fig_nb_5.png)

*Figura 8. Historia de cada punto llevada a 4 noches y predicción para la próxima visita con su intervalo del 80%.*

### 10. Otras preguntas y qué mediríamos

Nuestra pregunta fue si las capturas de toda la región suben y bajan juntas. Lo hacen: el promedio de cada ronda explica el 23% de lo que se aleja cada campaña del promedio de su sitio, cuando por azar esperaríamos un 6%. Pero no se mueven todos igual. Los sitios de An. albimanus y los de An. neivai siguen curvas distintas (correlación 0.06): en septiembre de 2021 cayeron los de albimanus y en marzo de 2022 los de neivai. Nuestra lectura, que es una inferencia, es que algo común en cada mes, como la lluvia, afecta distinto a cada especie.

La medición que agregaríamos es la lluvia de los días anteriores a cada noche, con un pluviómetro en la vereda o la estación del IDEAM más cercana. Con estos datos no se puede saber qué pasa hora por hora, cuánto se pica dentro contra fuera, si siempre se muestreó la misma vivienda, qué intervención recibió cada conglomerado ni cuántos mosquitos estaban infectados.

### 11. Lo que cambió en el camino

| Qué cambió                                                                | Por qué                                                                                                                        |
|-------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| De trampas a horas de recolector                                              | El estudio capturó con atrayente humano. En horas nominales el esfuerzo coincide con el de nuestra Fermi                           |
| De dónde poner trampas a cuántos mosquitos trae cada punto                    | La tabla no trae dentro o fuera, hora ni distancia al agua, y en Cheté hay solo dos puntos                                         |
| La separación de 2023 se hizo antes de elegir los datos                       | Para que la elección no mirara 2023                                                                                                |
| 2023 pasó de validación a prueba final, y la validación es 2022 ronda a ronda | Las campañas de un mismo sitio no son independientes                                                                               |
| De todo el estudio a Timbiquí                                                 | Los municipios difieren en especie y en qué pesa más. Se decidió después de una primera versión que ya mostraba resultados de 2023 |
| Corregimos el año de seis campañas de 2023                                    | eventDate y verbatimEventDate no coincidían. Lo encontramos después de varias ejecuciones de la evaluación final                   |
| El mes y la lluvia no resultaron clave                                        | El mes explica el 7% y el conglomerado el 44%. La lluvia no está en la tabla                                                       |
| Del intervalo de la Poisson al intervalo con errores de validación            | Los conteos varían 141 veces más de lo que supone la Poisson                                                                       |
| De seis variables a dos                                                       | La selección hacia adelante en la validación de 2022. Se hizo cuando ya conocíamos resultados de 2023                              |
| El PCA quedó como descripción                                                 | Con 16 conglomerados no aporta al modelo                                                                                           |

### 12. Uso de la IA

Usamos un asistente de IA (Claude) para revisar el plan, buscar fallas en la limpieza, escribir y depurar el código y revisar la redacción. Las sugerencias que cambiaron decisiones fueron comparar los dos años de cada fecha (así apareció el error de 2023), revisar que el número de noches fuera el mismo para todas las especies de una campaña, usar una regresión de Poisson con el esfuerzo multiplicando, construir el intervalo con los errores de validación y hacer la selección de variables con una regla fijada de antemano. Todas las cifras se verificaron corriendo el notebook. El reto pedía no mostrarle a la IA las campañas reservadas, y eso no se cumplió: el asistente ejecutó el notebook completo, incluida la evaluación final, en varias versiones. Por eso la evaluación de 2023 no es ciega y la reportamos con esa reserva.
