# Contenido de example.py
import numpy as np
import matplotlib.pyplot as plt
from linreg import ajustar, hipotesis, costo
from sklearn.linear_model import LinearRegression
 
np.random.seed(0)
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2*X + 0.02*np.random.random(100)
 
# Ajustar con nuestra libreria
res = ajustar(X, y, alpha=0.5)
print(f"linreg -> theta0={res['theta0']:.6f}, theta1={res['theta1']:.6f}")
 
# Comparar con sklearn
sk = LinearRegression().fit(X.reshape(-1,1), y)
print(f"sklearn -> theta0={sk.intercept_:.6f}, theta1={sk.coef_[0]:.6f}")
 
# Grafico
plt.scatter(X, y, s=15, color='gray', label='Datos')
plt.plot(X, hipotesis(X, res['theta0'], res['theta1']), 'tomato', label='linreg')
plt.plot(X, sk.predict(X.reshape(-1,1)), 'steelblue', linestyle='--', label='sklearn')
plt.legend()  ;  plt.tight_layout()  ;  plt.show()
