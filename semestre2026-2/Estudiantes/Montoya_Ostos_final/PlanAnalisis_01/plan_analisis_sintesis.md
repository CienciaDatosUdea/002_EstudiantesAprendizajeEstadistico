# Síntesis del plan

Juan Montoya y Laura Ostos

## La idea y cómo la contrastamos

La [Fermi original](../EstimacionFermi_00/estimacion_inicial.md) estimó 3000 ± 800 mosquitos en cuatro noches. Propusimos cuatro frascos por noche, unas 8 horas, dos noches en población y dos cerca de la mina. Se propusieron noviembre o diciembre; el plan concretó noviembre. Los 200 ± 50 por frasco y noche daban 3200 ± 800 antes de redondear. Los [50–200 huevos por puesta descritos por los CDC](https://www.cdc.gov/mosquitoes/pdfs/AnophelesLifeCycle-ESP.pdf) fueron una referencia intuitiva, no una medición de adultos por frasco.

Tres decisiones: (1) una visita reúne los puntos de una localidad en el mismo mes; (2) predecimos hembras adultas de *Anopheles* por hora de colector; para planear cuatro noches multiplicamos por 96 (2 personas × 12 h × 4 noches); (3) comparamos el error absoluto medio (MAE) con dos reglas: tasa histórica y última visita. La reserva de enero–marzo de 2023 se abrió para evaluar Poisson. La comparación posterior de los bosques no fue otra prueba a ciegas.

## Las tres preguntas: revisión retrospectiva (solo desarrollo 2021–2022)

1. **¿Qué factores se midieron?** Punto, calendario y esfuerzo permiten comparar tasas. El ZIP no separa interior/exterior ni mide color, señuelo, ácido láctico, CO₂, lluvia o exposición minera. La [hoja revisada](plan_analisis.md#hoja-de-campo-de-mosquitos) distingue horas efectivas, total, *Anopheles* confirmados o pendientes y fallos: cero no es un dato faltante. Antes se priorizó la lluvia; ahora se prioriza registrar **CO₂ ambiental en ppm y hora**, separado del señuelo, por ser una hipótesis de la Fermi no medida. No identifica gases mineros ni causas; la lluvia sigue siendo necesaria para comprobar la hipótesis de temporada.
2. **¿Noviembre, diciembre y la noche?** En 2021 se observaron 1,99 por hora en noviembre y 0,69 en diciembre; en 2022, 3,75 y 1,30. Juntos quedaron por debajo del resto de meses en 2021 (1,56 frente a 1,99) y apenas por encima en 2022 (2,52 frente a 2,43). No hay ventaja consistente del bimestre ni datos para comparar día y noche.
3. **¿Qué especies esperábamos y cuáles quedaron registradas?** La Fermi contemplaba *Aedes*, *Culex* y *Anopheles*; citó *An. darlingi* y *An. albimanus* a partir de [Wikipedia, fuente 2](https://es.wikipedia.org/wiki/Anopheles), como antecedente, no inventario. El archivo completo del [INS](https://doi.org/10.15472/kftrcn) tiene 1502 filas de hembras adultas de *An. neivai*, *An. albimanus* y *An. apicimacula*. No registra rasgos físicos ni demuestra ausencia de otros géneros. En noviembre y diciembre de ambos años, *An. albimanus* dominó en Cheté (1471/1529) y *An. neivai* en el panel de **21** localidades externas (38 059/41 678). El panel anterior de **24** compara solo los dos noviembres.

**¿Se repite la diferencia entre puntos?** Chete1 superó a Chete2 en 18 de 19 visitas: diferencia media de 2,62 por hora (IC del 95 % aproximado: 1,79–3,62). Fuera de Cheté, el orden se mantuvo en 14 de 17 pares de 11 localidades. Al exigir cinco meses comunes y dos posiciones distintas e inequívocas, se mantuvo en 12 de 15 pares. Algunos comparten puntos; el resultado no identifica un efecto minero.

## Qué cifra usar

En los dos noviembres de Cheté hubo 1197 individuos en 432 horas de colector (2,77/h). El escenario de 355–443 en frascos supone 8–10 h por noche y el mismo rendimiento por hora que una persona (*q* = 1). No es un conteo observado: para estimar *q* habría que comparar ambos métodos.

La tasa histórica tuvo el menor MAE en las diez visitas de 2022 (0,89/h); repetir la última visita ganó en las tres de 2023 (0,47/h). Ningún bosque superó a ambas reglas en los dos períodos. Con las 22 visitas hasta marzo, la referencia para abril de 2023 es 2,08 por hora (rango orientativo del 80 %: 0,47–3,69), o unos 200 en 96 horas (45–354). No hay observación de abril para comprobarla ni margen validado para noviembre.

[Notebook](../SintesisFinal_02/RetoMosquitos.ipynb) · [Presentación final, 15 diapositivas](../SintesisFinal_02/Presentacion.pdf).
