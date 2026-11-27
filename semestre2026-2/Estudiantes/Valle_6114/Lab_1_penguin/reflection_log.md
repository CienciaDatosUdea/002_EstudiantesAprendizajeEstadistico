# Reflection log

## Iteración 1
- Hipótesis: Existe una relación lineal entre bill_length_mm y bill_depth_mm; la dirección y magnitud se evaluarán con Pearson.
- Código: `pearson(bill_length_mm, bill_depth_mm)`
- Resultado numérico: `{"n": 342, "r": -0.235053}`
- Objeciones: ["La variable de agrupación especie no está controlada; especies distintas tienen rangos y medias de pico diferentes.", "La muestra no es homogénea: mezcla Adelie, Chinstrap y Gentoo, y especie e isla están parcialmente confudidas.", "La conclusión no debe interpretarse como causalidad ni como relación representativa dentro de cada especie."]
- Bloqueante: `True`
- Qué cambió: Se estableció una línea base global con Pearson.

## Iteración 2
- Hipótesis: Existe una relación lineal entre bill_length_mm y bill_depth_mm; la dirección y magnitud se evaluarán con Pearson.
- Código: `pearson(bill_length_mm, bill_depth_mm) por especie`
- Resultado numérico: `{"Adelie": {"n": 151, "r": 0.391492}, "Chinstrap": {"n": 68, "r": 0.653536}, "Gentoo": {"n": 123, "r": 0.643384}}`
- Objeciones: []
- Bloqueante: `False`
- Qué cambió: Se repitió Pearson dentro de cada especie para atender heterogeneidad y confusión por grupos.
