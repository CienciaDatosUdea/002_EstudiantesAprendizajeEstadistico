# Contenido de README.md
# linreg - Regresion Lineal con Gradiente Descendente
 
## Instalacion
```bash
pip install -e .   # instalar en modo editable desde la carpeta del paquete
```
 
## Uso rapido
```python
import numpy as np
from linreg import ajustar, hipotesis
 
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2*X + 0.02*np.random.random(100)
 
resultado = ajustar(X, y, alpha=0.5)
print(resultado['theta0'], resultado['theta1'])
