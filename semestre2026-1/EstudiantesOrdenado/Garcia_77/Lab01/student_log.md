## Phase 0 - Pre-Analysis

### "¿De qué trata este dataset y qué representa cada fila en términos concretos?":

Es un dataset de pinguinos

sex                   sexo del pinguino
bill_depth_mm         profundidad del pico  
bill_length_mm        largo del pico
flipper_length_mm     longitud de aleta
body_mass_g           masa
island                isla donde se 
species               especie del pinguino




### "¿Qué te da curiosidad investigar en estos datos y qué patrones esperas encontrar?":

Quisiera saber las correlaciones entre los datos que tenemos a disposición, si hay alguna conección entre variables como el largo del ala y el sexo, o qué caracteriza cada especie según estas características del ala y del pico. También si la isla donde el pinguino fue estudiado influye en sus características.


### "¿Qué variables crees que podrían estar relacionadas y por qué lo piensas?":

sexo y longitud profundidad de pico y longitud de aleta. Lo pienso porque muchas especies pueden tener dimorfismos, por lo que ayudaría a caracterizar el sexo de un pinguino si no se sabe a priori con otras características.
   
###  "¿Qué resultado te sorprendería y qué resultado confirmaría tu expectativa inicial?":

Me sorprendería ver alguna correlación extraña por ejemplo con el lugar donde se tomó el dato (isla) con alguna de las variables, ya que podría implicar a que un cambio de condiciones puede afectar las caracterísitcas en una misma especie. Un resultado que confirmaría mi expectativa inicial es una corelación entre especies y características, y dimorfismo sexual (dependencia de las caracterísitcas según el sexo
).


## Phase 2 - Describe Reflection

1. Al parecer en la gráfica de largo de ala vs largo del pico se puede ver que cada especie tiende a estar sectorizada en una región diferente.

2. En las variables numéricas se ve una correlación fuerte entre la masa y el tamaño del ala; sin embargo, sería interesante segmentarlo por especie para caracterizar las correlaciones entre especie.

3. Creo que podríamos usar las pruebas básicas y ver qué pasa.