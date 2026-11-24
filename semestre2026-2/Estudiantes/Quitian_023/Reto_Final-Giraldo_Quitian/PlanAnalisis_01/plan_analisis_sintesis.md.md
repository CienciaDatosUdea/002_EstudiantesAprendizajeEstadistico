# Cuatro noches para contar mosquitos. Versión sintetizada

Felipe Quitian Gallego, Miguel Angel Giraldo

La pregunta era cuántos mosquitos traeremos después de cuatro noches en Cheté. Con 8 trampas, 6 horas, 4 noches y 1 mosquito por trampa y hora estimamos 192 (entre 96 y 576), y creíamos que lo más importante sería la cercanía al agua estancada.

Al abrir el dataset cambiamos de perspectiva. El estudio del INS es un ensayo de intervención con captura por atrayente humano, dos recolectores por punto de 17:00 a 5:00, y en Cheté solo dos puntos. La tabla no trae hora, sitio dentro o fuera, vivienda, distancia al agua ni la intervención de cada sitio, así que pasamos de preguntar dónde poner trampas a preguntar cuántos mosquitos trae cada punto en cada visita.

Revisamos las columnas (31 de 69 no cambian) y la calidad. Lo principal fue corregir el año de seis campañas de 2023 que aparecían como de 2022. Una campaña es la observación, y como las campañas duran entre 1 y 6 noches y la línea base tuvo noches de 14 horas, normalizamos con esfuerzo = noches × horas / 12 y tasa = total / esfuerzo, en escala logarítmica porque el ruido es multiplicativo. Faltan 3 campañas y ninguna tiene cero mosquitos.

Separamos las 48 campañas de 2023 como prueba final antes de elegir los datos, y usamos 2021 para entrenar y 2022 para validar ronda a ronda, porque las campañas de un mismo sitio no son independientes. Los tres municipios se visitaron en las mismas fechas, pero no pesan igual las variables y Guapi casi no tiene An. albimanus, la especie de Cheté. Elegimos Timbiquí, con 16 conglomerados.

Al segmentar, lo que más pesa es la ubicación: el conglomerado explica el 44% de la variación, la especie dominante el 16%, la fecha el 13% y el mes el 7%, mientras que las noches y las personas no se distinguen del azar. Los conglomerados siguen un gradiente geográfico de especie que el PCA resume pero que no aporta al modelo. Cheté no representa al municipio: es casi solo An. albimanus.

En Cheté los dos puntos juntos traen una mediana de 417 mosquitos en 4 noches. Nuestro esfuerzo coincidía con el del estudio, así que la Fermi falló por la tasa, 2.2 por recolector y hora y no 1 (2 × 2 × 12 × 4 × 2.2 ≈ 420). Chete 1 captura unas 3.8 veces más que Chete 2, a 227 metros, pero no en la línea base, y no podemos separar el sitio de la intervención.

Para predecir comparamos referencias sencillas, una regresión lineal y una regresión de Poisson, μ = esfuerzo · exp(θᵀx), entrenada con descenso del gradiente. La Poisson es la adecuada para conteos multiplicativos y nunca da negativos. La selección de variables dejó dos, el promedio histórico del punto y su última campaña. En la validación deja el 70% de las campañas dentro de un factor 2 y le gana a la lineal, pero no al promedio del sitio, y no hay sobreajuste: el límite es el ruido entre campañas. Su sesgo es acercar cada sitio al promedio. En 2023 el promedio del sitio quedó algo mejor (81% contra 79%) y el intervalo del 80% cubrió al 88%.

| Modelo                | Dentro de un factor 2, validación 2022 | Dentro de un factor 2, prueba 2023 | Error relativo, validación 2022 | Error relativo, prueba 2023 |
|---------------------------|--------------------------------------------|----------------------------------------|-------------------------------------|---------------------------------|
| Tasa típica de Timbiquí   | 0.49                                       | 0.52                                   | 63%                                 | 60%                             |
| Repetir la última campaña | 0.62                                       | 0.77                                   | 52%                                 | 48%                             |
| Promedio del sitio        | 0.66                                       | 0.81                                   | 52%                                 | 39%                             |
| Regresión lineal          | 0.61                                       | 0.77                                   | 50%                                 | 42%                             |
| Regresión de Poisson      | 0.70                                       | 0.79                                   | 48%                                 | 40%                             |

La tabla da la fracción de campañas que quedaron dentro de un factor 2 y el error relativo de cada modelo, de cada 100 mosquitos capturados en cuántos se equivocó. Que el acierto en entrenamiento (62%) no supere al de validación ni al de prueba, y que la curva de aprendizaje casi no mejore con más rondas, muestra que el modelo no memoriza. Lo que lo limita es la variación de una campaña a otra, que ninguna columna de la tabla explica.

Para la próxima visita de 4 noches predecimos 284 mosquitos en Chete 1 y 111 en Chete 2, unos 406 en total, con un 80% de probabilidad entre 196 y 825 y un 4% de superar 1000. Es el doble de nuestra Fermi, por la tasa. La cifra de Chete 2 probablemente es alta, y el promedio del sitio (322) es una cota baja. Los frascos los fija el protocolo (inferimos 96 por conglomerado, falta confirmarlo); la sílica gel y las horas de laboratorio sí dependen de esta cifra.

![](figuras/fig_nb_6.png)

*Figura 9. La figura central del trabajo: historia de los dos puntos de Cheté llevada a 4 noches, la diferencia entre Chete 1 y Chete 2, la caída de septiembre de 2021, y la predicción para la próxima visita con su intervalo del 80%.*

Nuestra pregunta mostró que las capturas de la región suben y bajan juntas (la ronda explica el 23% de lo que se aleja cada campaña del promedio de su sitio, 6% por azar), pero distinto para cada especie: en septiembre de 2021 cayeron los sitios de An. albimanus y en marzo de 2022 los de An. neivai. La medición que agregaríamos es la lluvia de los días anteriores a cada noche.

En el camino cambiamos de trampas a horas de recolector, de todo el estudio a Timbiquí, de 2023 como validación a 2023 como prueba final, del intervalo de la Poisson a uno construido con los errores de validación y de seis variables a dos, y corregimos el año de seis campañas. Las limitaciones principales son que no sabemos qué intervención recibió cada conglomerado, que solo hay dos rondas de línea base y que la evaluación de 2023 no fue ciega, porque una primera versión ya mostraba esos resultados y el asistente de IA los vio.
