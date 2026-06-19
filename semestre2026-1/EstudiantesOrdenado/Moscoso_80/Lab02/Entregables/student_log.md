# Student Log

> Fill in your Phase 0 pre-analysis here before running the agent.
> Presionar Alt+Z para leer bien (line-break)


# Parte 1: Series de tiempo

## Phase 0 —  Student Analysis                                               


1. ¿De qué trata el dataset? ¿Qué representa cada fila? (Piensa en qué es una 
serie de tiempo de tipo de cambio)

Cada fila es la actualizacion de la proporcion de intercambio entre las divisas EURO y USD (dolar americano) en un período específico. El dataset dado por ejemplo está en periodos de una hora (1H). Nos dan el precio de apertura, máximo. mínimo y cierre durante ese período, además de otras columnas que ofrecen los brokers como el tick_volume, que representa la cantidad de transacciones (intercambios) usando este par de divisas (la cantidad de veces que el precio cambió), spread que tiene que ver con la ganancia de los brokers (pips) y la diferencia en el cuarto decimal entre las proporciones de apertura y cierre, real_volume que hace referencia a la cantidad total de dinero que realmente se negoceó.


2. ¿Qué te genera curiosidad sobre estos datos? ¿Qué patrones esperas         
encontrar en una serie de tiempo financiera EUR/USD?

Con estos datos es con lo que hacen los gráficos típicos de fluctuación del mercado financiero (bolsa internacional) que utiliza el sistema de velas verdes y rojas para indicar si el mercado va a favor o en contra de cierta divisa. Me da curiosidad saber como funcionan estos sistemas que manejan volumenes de datos enormes sincronizados a milisegundos de reloj para evitar pérdidas o desastres financieros.


3. ¿Qué variables crees que están relacionadas? (Por ejemplo: precio de       
apertura vs cierre, volumen vs volatilidad, etc.)

Esta es difícil. Creo que solo estarán relacionadas las variables derivadas (calculadas) matematicamente a partir de otras. Afirmar que hay correlacion positiva/negativa entre "open" y "low" por ejemplo es ingenuo pues la esencia del mercado es fluctuar en el tiempo. Si una correlacion entre estas dos variables fuera constante en el tiempo, el mercado seria predecible y perdería un poco el sentido.

Me da la sensacion que una columna calculada como la diferencia entre open  y close sí podria estar relacionada con ya sea "tick_volume" o "real_volume".


4. ¿Qué te sorprendería? ¿Qué confirmaría tus expectativas? (Por ejemplo:     
¿esperarías tendencias, ciclos, o comportamiento aleatorio?)

Pues el dataset contiene datos entre las fechas: "2022-07-25 13:00:00" y "2023-05-12 23:00:00". Casi un año. Esperaría encontrar grandes subidas o grandes depresiones en el precio de intercambio de estas dos divisas, e intentaría buscar noticias de ese día para dilucidar si hay correlacion con algun evento o noticia importante que haya afecta el mercado y el precio del par. Al fin y al cabo, las fluctuaciones y la esencia estocástica de estos mercados se debe a la impredicibilidad humana y las cosas que suceden en el día a día en muchísimos ámbitos. No quiero decir que espero encontrar ciclos. A lo mejor puede haber una tendencia negativa, porque en 2022 fue el año en el que la IA empezó a tomar fuerza, y el lider en ese entonces (ahora es discutible) era Estados Unidos. Pero en general, espero un comportamiento aleatorio.¿


## Phase 1 — OBSERVE

1. El dataset ya trae 4 columnas derivadas: MeanCloseOpen, Diff_Close, Diff_Open, Diff_MeanCloseOpen. Antes de calcular nada, ¿qué crees que representa cada una? ¿Cuál de estas se acerca  más a lo que intuiste en tu pre-análisis (Phase 0) cuando dijiste que la diferencia entre open y close podría relacionarse con el volumen?

Creo que MeanCloseOpen es la más cercana a la diferencia entre open y close, solo que debe estar promediada de alguna forma. Las otras columnas calculadas tengo la intuicion de que deben ser diferencias entre periodos consecutivos.

2. La columna real_volume tiene exactamente 1 valor único en las 5,000 filas. ¿Qué implica eso para cualquier análisis que intente usarla? ¿Deberías incluirla como feature o descartarla, y por qué?                                                                                                     

Debe ser descartada. Una columna con un valor constante no marca ninguna diferencia para el entrenamiento de un modelo o para un análisis como tal. A lo mejor es una columna estandar de este tipo de datasets pero fue censurada a propósita.                                                                 


3. La columna time fue detectada como tipo str en lugar de datetime. ¿Qué operaciones propias de una serie de tiempo se volverían imposibles o incorrectas si dejas time como string? ¿Qué deberías hacer en Phase 2 (Clean) para corregirlo?

No podría calcular la diferencia entre fechas por ejemplo, un calculo fundamental para obtener la periodicidad de los datos. Para corregir esto se podría usar por ejemplo la función pd.to_datetimne(df['time']) de pandas.


4. No hay valores faltantes ni duplicados. Para un dataset de trading descargado de un broker, ¿esto te parece esperado o sospechoso? ¿Podría haber "huecos" en la serie de tiempo que el reporte de missing no detectaría (por ejemplo, horas donde el mercado estuvo cerrado)?

Se debe hacer ese análisis cuidadosamente a parte. El runner claramente se limita a análisis generales basado en los tipos de variables. Pero es cierto, no hay una continuidad completa en los intervalos de tiempo de 1H porque los mercados cierran y abren todo.


## Phase 2 - Describe

1. Mira el heatmap: open, high, low, close y MeanCloseOpen tienen correlaciones de Pearson ≈ 1.00 entre sí. ¿Qué te dice eso sobre la información que aporta cada una de estas columnas de forma independiente? Si quisieras construir un modelo predictivo, ¿usarías todas ellas como variables de entrada?

La informacion que aportan de forma independiente es equivalente (pero no exactamente la misma). Me parece un comportamiento extraño que realmente solo esperaria entre por ejemplo "open" y "MeanCloseOpen" al ser esta última una columna derivada (a partir de un cálculo matemático) de las columas Open y Close. Sin embargo, puedo entender que exista un patrón en todas las filas de estas columnas. El único que no me cuadra y no logro entender por qué es la correlacion entre open y close, pues se supone que estos dos son los datos de comportamiento estocástico mas granulares que se tiene en este tipo de datasets. Definitivamente no las usaria todas ellas como variables de entrada, puede ser redundante e introducir sesgos.


2. La variable id tiene correlación ≈ 0.86 con las variables de precio (open, high, low, close). id es simplemente un índice secuencial (fila 1, 2, 3…). ¿Qué implicación tiene que el índice de tiempo esté tan correlacionado con el precio? ¿El EUR/USD tiene tendencia en este periodo? ¿Sube o baja?

Debe ser una mera casualidad pero lo que plantea la pregunta debe ser cierto. Si se hace un plot de, por ejemplo, MeanCloseOpen vs Tiempo probablemente se tendrá una tendencia de crecimiento positivo. Lastimosamente este tipo de correlaciones no se pueden generalizar en estos tipos de datos a no ser que se descubra un patron de "temporadas" con suficientes (muchos) datos para poder aseverar algo.


3. Las columnas Diff_Close, Diff_Open y Diff_MeanCloseOpen — que representan cambios período a período — tienen correlación ≈ 0.00 con las variables de nivel (open, high, low, close). Sin embargo, entre ellas se correlacionan con ≈ 0.70. ¿Por qué crees que la 'diferencia' de una serie y su 'nivel' pueden ser casi independientes entre sí? ¿Qué concepto estadístico podría estar detrás de esto?

Es una muy buena pregunta. La respuesta corta es no lo sé. Pero puedo hipotetizar que las columnas de diferencia entre periodos estan conectando dos "eventos" independientes. Mientras que las variables de nivel pues estan reservadas a un solo periodo de tiempo. Lo que sí no me explico es por qué no hay correlacion positiva considerable entre "DiffClose" y "DiffOpen" pero sí entre "Close" y "Open". No sé si la misma justificacion aplica aquí.

      
4. Ahora observa el histograma del spread: la gran mayoría de las velas tienen spread muy bajo (primer bin), pero hay un grupo secundario concentrado alrededor de 25–40, y valores aislados que llegan hasta ~175. ¿A qué eventos del mercado forex atribuirías ese cluster secundario? ¿Qué pasaría si usaras el spread como variable en un modelo sin revisar estos outliers primero?

Despues de consultar rapidamente sobre el tema se me ocurre que son dos rangos transaccionales que manejan los brokers, me imagino que dependiendo de la cantidad conversada (el monto del contrato) dependerá la cuota de spread entre compra/venta que usa el broker para tramitar las divisas. Como se ve en el histograma, hay valores de spread muy muy estandarizados ya para la gran mayoria de transacciones (rango de spread de 0-10 con la gran mayoria de frecuencia). Yo considero que se debe tratar con cuidado esta columna para que trate los casos outliers del spread de forma especial. Por ejemplo, se podria hacer un análisis de qué otras caracterísiticas diferentes tienen aquellos registros del dataset con valores de spread > 30, e intentar agruparlos.



## Phase 3 - Pre-Hypotheses

1. En Phase 0 dijiste que esperabas 'comportamiento aleatorio' pero en Phase 2 encontramos que id (tiempo) y precio tienen correlación de Pearson ≈ 0.86. Ahora que tienes ese dato: ¿cuál es tu hipótesis sobre la dirección del EUR/USD en el período jul 2022 – may 2023? ¿Sube o baja? Formula eso como una hipótesis falsificable: 'El precio de cierre del EUR/USD [aumentó / disminuyó] de forma sostenida durante el período, lo que se evidenciaría en...'",

Viendo el id como una columna de temporalidad, sí podemos hipotetizar: "La tasa de cambio entre EUR/USD en el período jul 2022 - may 2023 tuve una temporada de crecimiento". Para validar esto se debe hacer una prueba de hipótesis nula.

2. La correlación entre tick_volume y las variables de precio es ≈ -0.28 a -0.31. En Phase 0 intuiste que las diferencias (Diff_*) podrían relacionarse con el volumen. Ahora que ves que es el *nivel* de precio — no la diferencia — el que se asocia negativamente con el volumen: ¿qué hipótesis te genera eso? ¿En qué tipo de período (precio alto vs precio bajo) esperarías ver más actividad transaccional?

La correlacion negativa no es demasido considerable pero sí podemos formular la hipótesis: "Cuando la tasa de intercambio EUR/USD aumenta, el volumen de transacciones tiende a disminuir".


3. El histograma del spread muestra dos clusters claramente separados: uno dominante cerca de 0 y uno secundario alrededor de 25–40. ¿Qué hipótesis te genera sobre *cuándo* ocurren esas velas con spread alto? ¿Estarían concentradas en ciertos momentos del día, días de la semana, o períodos de precio inusual? Formula una hipótesis que el runner podría ayudarte a explorar."

Para formular una hipótesis en este aspecto, primero intentaria observar el patron de temporalidad ploteando MeanCloseOpen vs Time. Intentaria buscar por periodos en los que el patrón cambia o reinicia su ciclo. A lo mejor, al final/inicio de trimestres o semestres donde las empresas entregan reportes financieron y hay cambios importantes en el mercado. Sin embargo, aunque no soy experto en estos temas, siento que el "spread" debe estar mas relacionado con el monto neto de las transacciones (valor al que no tenemos acceso directo). Y a lo mejor, este tipo de transacciones "grandes" suceden en periodos de tiempo muy específicos.



# Parte 2: Análisis de datos con dataset de breast_cancer

## Phase 1 — Observe

**Dataset:**

1. "The artifact `02_missing_report__OBSERVE__20260314_103022.json` reports 0 missing values across all 32 columns and all 569 rows. Is this what you expected for a clinical dataset? What are the implications for the choices you will need to make in Phase 2 (e.g., imputation, dropping rows)?"

Como se puede leer en la documentación del dataset, estos datos ya han sido limpiados, curados (ajustados a un propósito) y seleccionados. Es por esto que no es de extrañar que no tenga datos nulos. La pregunta es válida, esto no es nada común si estuvieramos tratando data cruda de pacientes clínicos. Lo más común es que hayan datos faltantes regados entre los diferentes pacientes.

En este caso, para la fase 2 de imputacion o descarte de filas, no será necesario puesto que el dataset ya está bien limpio y listo para el análisis.


2. "Look at `01_schema__OBSERVE__20260314_103022.json` for the `id` column: it has 569 unique values out of 569 rows. What does that tell you about its role in any future analysis? Should it be treated as a feature, a target, or neither?"

Las columnas de ids nunca deberian ser consideradas features o targets dada su alta cardinalidad. Creo que su utilidad está más en su organizacion dentro de una base de datos (usar los ids para relacionar con otros objetos) o para identificar si no hay registros duplicados, asociados al mismo paciente tal vez. Pero para entrenar un modelo de ML o hacer análisis propiamente, los ids son descartables.



3. "The 30 numeric columns are organized in three groups of 10: `_mean`, `_se`, and `_worst`. Before running any statistics, what do you expect the relationship between, say, `radius_mean` and `radius_worst` to look like? Would you expect them to be strongly associated, weakly associated, or unrelated — and why?"

Están completamente relacionados. Segun la misma documentación sobre como se calcula el "_worst" de una feature dice:

"The mean, standard error, and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features."

Si una feature es producto de un cálculo de otras dos, entonces esta tendrá una relación directa (o indirecta) con estas.



##  Phase 2 - Describe

3. "Open `box_radius_mean_by_diagnosis`. The two box plots (B and M) for radius_mean — do they overlap, and if so, how much? What does that overlap tell you about how reliable a single feature would be for separating the two diagnoses?"

Dentro del rango intercuartílico, las características de "radius_mean" agrupado por "diagnoses" NO se sobreponen. Uno podría pensar que esta es entonces una buena feature, suficiente para separar los grupos y determinar el diagnóstico de un nuevo paciente, pero recordemos que hay barras de errores y outliers que nos indican que hay datos fuera del rango intercuartílico que se podrian confundir bajo esta característica. En general, no es fiable, aunque la característica separe bien los grupos aparentemente, depender de una sola variable para definir algo drástico como un diagnóstico de cancer. 


4. "Open `heatmap_corr_pearson`. The 30 features are ordered as: 10 _mean, 10 _se, 10 _worst. Do you see block structure along the diagonal? Now look off-diagonal: is the _mean block more correlated with the _worst block or the _se block? What might that suggest about what _worst and _se are measuring relative to _mean?"

La mayor correlación se encuentra entre los grupos de "_mean" y "_worst", confirmando lo que intuiamos en la fase 1 cuando discutimos que los datos en _worst son producto de una selección/cálculo sobre los valores de "_mean". Por supuesto el bloque de "_se" (errores estandar) tambien comparten correlación positiva moderada con los otros 2 bloques.


5. "The numeric summary shows that `fractal_dimension_mean` has a *negative* Pearson correlation with `radius_mean` (-0.312) and `area_mean` (-0.283), while most shape features correlate positively with size. Does this pattern make sense to you? What might it mean for a tumor to have large area but low fractal dimension — or small area but high fractal dimension?"

Para esto tuve que investigar un poco la documentacion del dataset. Luego de entenderla, tiene todo el sentido que "fractal_dimension_mean" tenga correlacion negativa con diversas variables numéricas del dataset. Siendo puntual con "radius_mean" y "area_mean" vemos en los histogramas que valores más altos de estas medidas están siempre asociados a tumores (M) Malignos. Por otro lado la dimensión fractal se mide en una escala inversa: entre más cercano a 0 significa más redondez y simplicidad tiene la frontera o el perímetro del tumor (común en tumores (B) Benignos), mientras valores más positivos indican una frontera más caótica, con más picos y partes rugosas, característicos de tumores (M).



## Phase 3 — Pre-Hypotheses

6. "You now have box plots for both `radius_mean` and `concave_points_mean` split by diagnosis, and histograms for `area_mean`, `area_worst`, and `concavity_mean`. Without looking at any numbers: which single plot shows the cleanest visual separation between M and B? Which shows the most overlap? Based on only what your eyes tell you, rank these four features by how useful you think they would be for distinguishing the two groups."

A priori una de las características que mejor diferencian ambos diagnósticos es "Radius_mean" (Viendo el box_radius_mean_by_diagnoses, noto que los rangos intercuartílicos estan bien separados y solo hay unos pocos outliers con alto radius_mean en el grupo B que se cruzan con el rango I.C del grupo M), pero definitivamente un diagnóstico no se puede hacer basado solo en esta.

Luego estarían area_worst y area_mean, ambas se distribuyen de formas muy similar, con colas considerables hacia los valores altos para el grupo M, mientras que los grupos B se quedan en valores bajos.

Luego colocaría concavity_mean pues su rango de valores es estrecho, aunque viendo su distribución es claro que los del grupo B se concentran usualmente en valores bajos (muy alta frecuencia), pero hay unos cuantos outliers.

Y por último dejaría concave_points, pues viendo su scatter con concavity_mean, uno podría imaginarse una frontera, pero realmente hay muchos datos sobre la frontera difusa. En su box plot agrupado por diagnóstico tambien vemos la alta cantidad de valores outliers que hay.




7. "The Pearson correlation matrix shows that `radius_mean`, `perimeter_mean`, and `area_mean` all correlate with each other at r > 0.985. If you were building a model that used all 30 features, what problem would this extreme redundancy create? Which of the three would you keep, and why?"

Tiene sentido esa correlación. Todas estan estrechamente conectadas e incluirlas en el entrenamiento de un modelo sería introducir un sesgo (3 pesos "independientes" para valores que son realmente dependientes). Para reducir dimensionalidad y disminuir este sesgo, usaria la que en un box plot agrupado por diagnóstico me permite diferenciar mejor los grupos (los separe mejor).


8. "The scatter plot `scatter_concavity_mean_vs_concave_points_mean_hue_diagnosis` shows both features together, colored by diagnosis. Based on what you see (or expect to see given their r = 0.921), do you think a straight line could reasonably separate M from B in that 2D space? What would that imply about the difficulty of this classification problem?"

Es reduccionista y delirante construir esa frontera con 2 características que estan estrechamente correlacionadas. Si mi frontera es una linea recta en un plano donde se representan dos características en un contexto médico como este, me parecería un modelo demasiado simple (underfit) para clasificar datos de alta importancia como un diagnóstico de cancer. Definitivamente mi criterio sin ver el resultado de dicho modelo sería que le falta complejidad al modelo.



