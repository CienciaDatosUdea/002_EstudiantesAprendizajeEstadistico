### Estimación de Fermi

En una aproximación inicial suponemos un módelo de captura basada en puestos de avanzada, es decir, puestos fijos donde nuestro equipo estara capturando activamente mosquítos durante un periodo de tiempo determinado. Por tanto el número de capturas será establecido como la suma de las capturas totales realizadas en cada puesto durante el intervalo de tiempo establecido

$$ N = \sum_i^M n_i$$

siendo $N$ el número de capturas y $n_i$ las capturas en el puesto i-ésimo. Las capturas en el puesto i-ésimo se pueden a su vez descomponer como una razon de capturas por el tiempo de captura, es decir

$$ n_i = H \times v_i  $$

$H$ se refiere al número de horas de captura y $v_i$ es la velocidad de capturas por hora. Suponiendo de modo simplificado ciertos puestos equivalentes, es decir con misma velocidad de capturas

$$ N = \sum_n P_n H v_n $$

donde $P_n$ es el número de puestos equivalentes de la clase $n$. 

En un módelo un poco más realista, suponemos que en realidad el número de capturas por puesto es una variable aleatoria de tipo Poisson, lo cual tiene mucho sentido por el tipo de experimento que estamos realizando, es decir cúal es el número de capturas que esperamos en un intervalo de tiempo $H$ dado una cantidad media de capturas por hora $\lambda$. Para realizar nuestra estimación empleando este modelo podemos emplear python y generar números aleatorios que sigan la distribución deseada (aproximada) y determinar la media y la desviación estandar tras tantas capturas.

Ambos modelos se pueden extender a capturas totales en un número $D$ de días de capturas.

Como hipotesis de nuestro modelo, supondremos dos tipos de sitio, Interiores (referentes a capturar dentro de una vivienda) y Exteriores, capturar en el exterior de una casa. Suponemos que la presencia de humanos a las horas de la noche aumentara la cantidad de capturas en el interior respecto al exterior. Así pues para el primer módelo 

$$ N = D \times H ( P_Iv_I + P_E v_E ) $$

supondremos un 

y para el segundo 

$$ N = \sum_d^D ( n^{(I)}_d + n^{(E)}_d ) $$

donde $d$ es el día, $n^{(I)}$ es un entero de Poisson con media $\lambda_I = 2 \times H$ y $n^{(E)}$ es otro entero de Poisson con media $\lambda_E = 1 \times H$. Suponiendo una situación bastante pesimista, con una tasa de captura de 2 y 1 mosquito respectivamente de media por hora.


### Estimación Númerica

Suponiendo que $H = 6$ y $D = 4$ y 3 puestos de cada tipo:

1) Primer Modelo: Suponiendo entonces 2 mosquitos por hora en interior y 1 por hora en exterior el primero modelo predice que se espera un total de $N = 6 \times 4 \times 3 \times (1 + 2) = 216$

2) Segundo Modelo: El segundo modelo tras $10.000$ runs entrego estas predicciones

| Estadístico | Columna 1 | Columna 2 | Columna 3 |
| :--- | :---: | :---: | :---: |
| **count** | 10000.000000 | 10000.000000 | 10000.000000 |
| **mean** | 143.987800 | 71.921600 | 215.909400 |
| **std** | 12.157526 | 8.602724 | 14.996456 |
| **min** | 103.000000 | 42.000000 | 159.000000 |
| **25%** | 136.000000 | 66.000000 | 206.000000 |
| **50%** | 144.000000 | 72.000000 | 216.000000 |
| **75%** | 152.000000 | 78.000000 | 226.000000 |
| **max** | 194.000000 | 110.000000 | 282.000000 |

