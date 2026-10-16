# lineareg

Libreria de Python para ajustar una regresion lineal de una variable mediante la funcion de coste cuadratica y gradiente descendente. La unica dependencia de ejecucion es `numpy`; el gradiente se calcula directamente en `modelo.py`, sin `scipy`, `sklearn` ni otras librerias.

## Instalacion

Desde la carpeta raiz del proyecto:

```bash
python -m pip install -e .
```

La instalacion editable permite importar `lineareg` mientras se modifica el codigo fuente.

## Uso

```python
import numpy as np
from lineareg import ajustar_modelo, hipotesis_lineal

X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X
modelo = ajustar_modelo(X, y, tasa_aprendizaje=0.1)
print(modelo["intercepto"], modelo["pendiente"])
print(hipotesis_lineal(0.5, modelo["theta"]))
```

La funcion `hipotesis_lineal` calcula `theta_0 + theta_1 * X`. `coste` calcula el error cuadratico medio dividido entre dos. `descenso_gradiente` retorna los parametros y el historial del coste. `ajustar_modelo` es la funcion principal y retorna un diccionario con los parametros y el historial.

No es necesario crear un entorno virtual para probar esta entrega. Solo se requiere tener Python y NumPy disponibles; el comando de instalacion anterior registra el paquete local en modo editable.

## Ejemplo del laboratorio

Despues de instalar la libreria, ejecute:

```bash
python examples/ejemplo_lab4.py
```

El ejemplo usa los datos del laboratorio, muestra el intercepto, la pendiente, el coste final y una prediccion.
