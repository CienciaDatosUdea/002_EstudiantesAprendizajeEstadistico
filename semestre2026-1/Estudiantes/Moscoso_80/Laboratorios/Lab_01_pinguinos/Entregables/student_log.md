##### Leer en VSCode con alt+Z para line-break automático

## Phase 0 — Pre-analysis

### "¿De qué trata el dataset de pingüinos? ¿Qué representa cada fila — un individuo, una especie, una observación de campo?"

El dataset contiene 344 filas y 7 columnas. Las columnas son : ['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']. Al parecer, propiedades que caracterizan a un pingüino específico, por lo tanto, tenemos 344 registros de pingüinos que incluyen datos sobre su fisionomía, lugar de origen y especie.


### "Antes de ver cualquier número: ¿qué patrones esperas encontrar? Por ejemplo, ¿crees que el tamaño del pico varía entre especies? ¿Por qué?"

El hecho de que haya diferenciacion entre islas me hace pensar que pudieron haber evolucionado de manera diferente. Me recuerda a mi clase de genética en el colegio, en donde diferentes especies de aves tenían picos diferentes para adaptarse a la naturaleza propia de las semillas de la isla donde solían habitar. Algo similar podría pasar con estas especies de pingüinos donde en cierta isla, la dieta de alimentos disponibles y formas de vida les haya hecho tener fisiologías diferentes. Sin embargo, creo que no necesariamente estarán diferenciados completamente por la isla. Es posible que una misma isla sea compartida por dos o más especies de pingüinos dado que no creo que ellos sean agresivos entre ellos.

### "¿Qué variables intuyes que están relacionadas entre sí? ¿Cuál sería la relación que más te sorprendería encontrar (o no encontrar)?"

Una correlación que intuyo que existirá será la de masa corporal y sexo. Creo que es común en la naturaleza (específicamente en mamíferos) que los machos tengan más masa corporal que las hembras. Lo contrario me sorprendería. A lo mejor también puede haber una correlación positiva entre la longitud de las aletas y la masa corporal, pues estas proporciones suelen ir de la mano.


### "¿Qué te parecería un resultado 'esperado' y qué te parecería sorprendente? Escríbelo antes de ver los datos — eso es lo que hace un científico."

Es difícil dilucidarlo sin saber mucho de pingüinos y tampoco ver los datos. Se me ocurre que tal vez, en una isla donde solo haya una especie de pingüinos (que no sé si sea el caso en el dataset), los pingüinos pueden aprovechar mejor los recursos y ser más fornidos (mayor masa corporal agrupando por sexos y comparando con otras especies de otras islas, mayor longitud de aletas, picos tal vez más robustos (aunque esto depende más es de las necesidades de su dieta, lo cual no está disponible en el dataset).

## Phase 1 — OBSERVE

### "El reporte de faltantes muestra que `sex` tiene 11 valores ausentes (3.2%) — más del doble que cualquier otra variable. En tu pre-análisis dijiste que esperabas una correlación entre `sex` y `body_mass_g`. ¿Qué implicación tiene este patrón de datos faltantes para esa hipótesis? ¿Crees que los registros sin `sex` son aleatorios o podrían corresponder a un tipo particular de pingüino?"

Podrian ser aleatorios o tener alguna correlacion con las otras variables. Valdria la pena explorarlo pero considero que 11/344 datos faltantes para la columna `sex` es poco. Si nos interesara explorar esa relacion entre `sex` y `body_mass_g` podriamos:

1. Eliminar las filas con datos faltantes en alguna de estas dos columnas.
2. Hacer imputación de datos usando alguna medida de tendencia central, dependiendo de la distribución de cada columna.


### "Las 4 variables numéricas (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`) tienen exactamente 2 valores faltantes cada una — el mismo número. ¿Qué te sugiere eso? ¿Podrían ser las mismas 2 filas con datos incompletos en todas las variables numéricas a la vez?"

Sí, eso podría sugerir que hay registros que son inconsistentes en todas estas columnas. Aunque no se descarta que esten en diferentes pingüinos. Habria que encontrar esos registros con una máscara en el dataframe.


## Phase 2 — DESCRIBE

### "Abre el box plot `box_bill_depth_mm_by_species`. La correlación global Pearson entre `bill_depth_mm` y `bill_length_mm` es -0.235 (negativa). Pero si miras los tres grupos por separado en el scatter `scatter_bill_length_mm_vs_flipper_length_mm_hue_species`, ¿ves el mismo signo negativo dentro de cada especie, o el patrón cambia? ¿Qué implicación tendría eso para analizar correlaciones sin separar por especie?"

En el scatter de bill_length vs flipper_length agrupados por especie, se puede hacer la separación clara entre las especies y además, la correlación entre estas variables para las 3 especies es positiva. Los grupos no se superponen demasiado y es posible dibujar perímetros que separe cada grupo. Claro está, hay unos pocos pinguinos que se cruzan entre los grupos de especie, pero no es tan notorio como en el scatter de bill_depth vs flipper_length donde los pingüinos de Chinstrap y Adelia se superponen casi completamnete (por lo tanto bill_depth y flipper_length no son características diferenciadoras entre estas dos especies).

Por otro lado, el hecho de que la correlacion entre bill_depth y bill_length sea negativa, no lo podemos relacionar con los scatter plot que comparan estas variables respecto a flipper_length. Para ver si el coeficiente de Pearson de -0.235 es consistente, deberiamos hacer un scatter plot de bill_depth y bill_length.


### "El resumen categórico muestra que `Adelie` aparece en 152 registros y `Chinstrap` en solo 68. Si quisieras comparar `body_mass_g` entre las tres especies, ¿este desbalance importa? ¿Qué tipo de prueba estadística sería más robusta ante grupos de tamaños distintos?"

Sí, veo que en el reporte de variable categóricas hay un desbalance claro entre los valores únicas tanto de especie como de isla. Pero las proporciones (y su desbalance) es muy similar tambien (esto no implica que podamos relacionar especia <-> isla). Tambien noté que el sexo, a pesar de ser la variable con más valores nulos, estaba bien balanceada.

Ahora, para comparar la variable body_mass_g entre las especies, considerando el desbalance, se me ocurre tal vez aplicar el método de Bootstraping agrupado por especies, escogiendo muestras aleatorias con reemplazo donde se toma la misma cantidad de datos de muestra de ambas especies a comparar. Pero no estoy seguro si sea la mejor estrategia.

### "La correlación Pearson entre `flipper_length_mm` y `body_mass_g` es 0.871 — la más alta del dataset. Tú anticipaste en Phase 0 que estas dos variables estarían relacionadas. ¿Esta magnitud confirma o supera tu expectativa? ¿Qué forma esperarías ver en el scatter de estos dos variables?"

Imaginaba una correlación pero no tan alta. Esperaba un 0.6-0.7 máximo. Sin embargo, no me parece anormal pues tiene todo sentido. Un pingüino más robusto (con más masa) puede tener una longitud de aleta más larga. 

En el scatter podría esperar ver una tendencia positiva, como una recta con pendiente alta. Los datos podrian estar dispersos porque podrían haber casos de pingüinos pequeños (aleta corta) pero con gran masa (pequeños y gordos). Pero según el coeficiente de Pearson, la mayoria de los datos cumplen esta correlación  muy positiva.


## Phase 3 — Hipótesis propias

### "Observaste que en el scatter `bill_length_mm` vs `flipper_length_mm` los tres grupos de especie son separables, pero en `bill_depth_mm` vs `flipper_length_mm` Chinstrap y Adelie se superponen. ¿Qué hipótesis falsable formularías sobre cuál variable morfológica diferencia mejor a las especies? ¿Cómo la probarías?"

Analizando:
- Histograma bill_length por especie
- Histograma de flipper_length por espcie
- Scatter plot de bill_length vs flipper_length

Puedo aseverar que estas dos variables numéricas son las mejores y suficientes para separar bien un individuo de cualquiera de las 3 especies. Es posible ver que las fronteras de los "grupos" en el scatter plot no son muy definidas, y un individuo con combinacion de estas variables sobre estas fronteras podría ser mal clasificado.

De forma reduccionista mi hipótesis se podría plantear como:
- Una combinación de bill_length pequeño y flipper_length pequeño es más probable que sea de un pingüino de la especie Adelie
- Una combinación de bill_length alto y flipper_length pequeño es más probable que sea de un pingüino de la especie Chinstrap
- Una combinación de bill_length alto y flipper_length alto es más probable que sea de un pingüino de la especie Gentoo


### "Anticipaste en Phase 0 que los machos tendrían mayor `body_mass_g` que las hembras. Ahora sabes que `sex` está casi perfectamente balanceado (168 M vs 165 F) y que `body_mass_g` tiene skewness de 0.47 con una cola hacia valores altos. ¿Crees que esa cola podría estar asociada a los machos? Formula esto como una hipótesis falsable con variables concretas."

Viendo el histograma de la variable body_mass agrupado por especie puedo ver que los rangos de masa entre especies son diferentes. En general, los pingüinos Gentoo son más masivos que Adelie y Chinstrap. Para intentar explicar esa valor de skewness hacia los valores altos de masa, yo investigaria como se distribuyen los sexos entre especies. Aunque la variable 'sex' esté balanceada, sabemos del análisis de variables categóricas que 'species' no lo está:

"value_counts": {
      "Adelie": 152,
      "Gentoo": 124,
      "Chinstrap": 68
    }

Esta desproporcion puede explicar esta asimetria en la distribucion de masa. Para validarlo, se debe ver como se distribuye el sexo de los pinguinos en las diferentes especies. Yo esperaria encontrar que hay mas pingüinos machos que hembras en la especie Gentoo (que es la más masiva), contribuyendo así a la asimetria del histograma de body_mass


### "Notaste que las proporciones de `species` e `island` son sorprendentemente similares (Adelie~44% / Biscoe~49%, Chinstrap~20% / Torgersen~15%). ¿Eso te sugiere que cada isla podría estar dominada por una sola especie? Formula una hipótesis sobre la relación entre `species` e `island` que pueda verificarse con una tabla cruzada o una prueba chi-cuadrado."

Se siente extraño plantear una correlación entre dos variables solo porque tienen proporciones similares dentro de sus valores únicos. Creo que eso está muy mal. Para poder aseverar esto es necesario hacer un análisis que involucre ambas variables a la vez. Pero no sé bien qué tipo de visualizacion permite esto.