# Lab 4

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from sklearn.linear_model import LinearRegression

# %% [markdown]
# # 1. Construya un `DataFrame` de pandas con un conjunto de datos lineales simples.


# %%
def df(m):
    Y = np.arange(m)
    X = np.arange(m)
    df = pd.DataFrame({"Y": Y, "X_1": X})
    return df


df = df(6)
df

# %% [markdown]
# # 2. Defina una función que calcule la función de coste cuadrática para un modelo de regresión lineal.


# %%
def func_coste(a0, a1, X, Y):
    m = len(X)
    return np.sum(((a0 + a1 * X) - Y) ** 2) / (2 * m)


# %% [markdown]
# # 3. Fijando inicialmente $\theta_0=0$, evalúe y grafique la función de coste para diferentes valores de $\theta_1$. Determine el valor que minimiza la función de coste y grafique la recta obtenida sobre los datos.

# %%
a0 = 0
a1_list = np.linspace(-5, 5, 100)
J = [func_coste(a0, a1, df["X_1"], df["Y"]) for a1 in a1_list]

plt.plot(a1_list, J)
plt.xlabel("$\\theta_1$")
plt.ylabel("Coste ($J(\\theta_1)$)")
plt.title("Función de coste")
plt.show()


min_index = np.argmin(J)
optimal_a1 = a1_list[min_index]
print(f"Valor óptimo de $\\theta_1$: {optimal_a1}")


def h(a0, a1, X):
    return a0 + a1 * X


# Graficar la recta obtenida sobre los datos
plt.plot(df["X_1"], df["Y"], "o", label="Datos")
plt.plot(df["X_1"], h(a0, optimal_a1, df["X_1"]), label="Modelo", color="red")
plt.xlabel("X")
plt.ylabel("h(X)")
plt.title("Datos y recta ajustada")
plt.legend()
plt.show()

# %% [markdown]
# # 4. Permita ahora que tanto $\theta_0$ como $\theta_1$ varíen. Construya una malla con `np.meshgrid`, evalúe la función de coste en cada punto y represente su superficie y curvas de nivel.

a0_list = np.linspace(-5, 5, 100)
a1_list = np.linspace(-5, 5, 100)
A0, A1 = np.meshgrid(a0_list, a1_list)
J = np.array(
    [[func_coste(a0, a1, df["X_1"], df["Y"]) for a0 in a0_list] for a1 in a1_list]
)

# Superficie
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(A0, A1, J, cmap="viridis")
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Coste ($J(\\theta_0, \\theta_1)$)")
ax.set_title("Superficie de la función de coste")

# Curvas de nivel
ax2 = fig.add_subplot(122)
contour = ax2.contour(A0, A1, J, levels=50, cmap="viridis")
ax2.set_xlabel("$\\theta_0$")
ax2.set_ylabel("$\\theta_1$")
ax2.set_title("Curvas de nivel de la función de coste")
plt.colorbar(contour, ax=ax2)
plt.tight_layout()
plt.show()

# %% [markdown]
# # 5. Interprete geométricamente la forma de la función de coste e identifique el mínimo global.

global_min_index = np.unravel_index(np.argmin(J), J.shape)
optimal_a0 = a0_list[global_min_index[1]]
optimal_a1 = a1_list[global_min_index[0]]
print(f"Valor óptimo de theta_0: {optimal_a0}")
print(f"Valor óptimo de theta_1: {optimal_a1}")


display(
    Markdown(
        f"Geométricamente, la función de coste es una superficie parabólica en el espacio de parámetros ($\\theta_0$, $\\theta_1$). El mínimo global se encuentra en el punto donde la función de coste alcanza su valor más bajo ($\\theta_0 = {optimal_a0}$, $\\theta_1 = {optimal_a1}$). En este caso, el mínimo global se encuentra cerca de $\\theta_0 = 0$ y $\\theta_1 = 1$, lo que indica que el modelo lineal ajusta bien a los datos con estos parámetros."
    )
)

# %% [markdown]
# # 6. Repita el procedimiento para un conjunto de datos con ruido y compare los resultados con el caso ideal.


# %% [markdown]
def df_noisy(m, noise_std=1.0):
    Y = np.arange(m) + np.random.normal(0, noise_std, m)
    X = np.arange(m)
    df = pd.DataFrame({"Y": Y, "X_1": X})
    return df


df_noisy = df_noisy(6)
df_noisy

# Evaluar la función de coste para el conjunto de datos con ruido
J_noisy = np.array(
    [
        [func_coste(a0, a1, df_noisy["X_1"], df_noisy["Y"]) for a0 in a0_list]
        for a1 in a1_list
    ]
)

# Comparar la función de coste con y sin ruido
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(A0, A1, J, cmap="viridis", alpha=0.5, label="Sin ruido")
ax.plot_surface(A0, A1, J_noisy, cmap="plasma", alpha=0.5, label="Con ruido")
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Coste ($J(\\theta_0, \\theta_1)$)")
ax.set_title("Comparación de la función de coste con y sin ruido")
ax.legend()

# Superficie para el conjunto de datos con ruido
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(A0, A1, J_noisy, cmap="viridis")
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Coste ($J(\\theta_0, \\theta_1)$)")
ax.set_title("Superficie de la función de coste con ruido")

# Curvas de nivel para el conjunto de datos con ruido
ax2 = fig.add_subplot(122)
contour = ax2.contour(A0, A1, J_noisy, levels=50, cmap="viridis")
ax2.set_xlabel("$\\theta_0$")
ax2.set_ylabel("$\\theta_1$")
ax2.set_title("Curvas de nivel de la función de coste con ruido")
plt.colorbar(contour, ax=ax2)
plt.tight_layout()
plt.show()


global_min_index_noisy = np.unravel_index(np.argmin(J_noisy), J_noisy.shape)
optimal_a0_noisy = a0_list[global_min_index_noisy[1]]
optimal_a1_noisy = a1_list[global_min_index_noisy[0]]
print(f"Valor óptimo de theta_0 con ruido: {optimal_a0_noisy}")
print(f"Valor óptimo de theta_1 con ruido: {optimal_a1_noisy}")


display(
    Markdown(
        f"Comparando con el caso ideal, el mínimo global para el conjunto de datos con ruido se encuentra en un punto diferente ($\\theta_0 = {optimal_a0_noisy}$, $\\theta_1 = {optimal_a1_noisy}$). Esto se debe a que el ruido introduce variabilidad en los datos, lo que puede afectar la forma de la función de coste y desplazar el mínimo global. En general, el modelo ajustado a los datos con ruido puede no ser tan preciso como el modelo ajustado a los datos ideales, lo que se refleja en los valores óptimos de los parámetros."
    )
)


# 7. Analice el efecto de introducir un valor atípico en el conjunto de datos. Discuta cómo cambia la solución y qué limitaciones presenta la función de coste cuadrática.

# Introducir un valor atípico en el conjunto de datos
df_outlier = df.copy()
df_outlier.loc[0, "Y"] = 30  # Introducir un valor atípico
df_outlier
# Evaluar la función de coste para el conjunto de datos con un valor atípico
J_outlier = np.array(
    [
        [func_coste(a0, a1, df_outlier["X_1"], df_outlier["Y"]) for a0 in a0_list]
        for a1 in a1_list
    ]
)

# Comparar la función de coste con y sin un valor atípico
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(A0, A1, J, cmap="viridis", alpha=0.5, label="Sin valor atípico")
ax.plot_surface(A0, A1, J_outlier, cmap="plasma", alpha=0.5, label="Con valor atípico")
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Coste ($J(\\theta_0, \\theta_1)$)")
ax.set_title("Comparación de la función de coste con y sin un valor atípico")
ax.legend()

# Superficie para el conjunto de datos con un valor atípico
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection="3d")
ax.plot_surface(A0, A1, J_outlier, cmap="viridis")
ax.set_xlabel("$\\theta_0$")
ax.set_ylabel("$\\theta_1$")
ax.set_zlabel("Coste ($J(\\theta_0, \\theta_1)$)")
ax.set_title("Superficie de la función de coste con un valor atípico")
# Curvas de nivel para el conjunto de datos con un valor atípico
ax2 = fig.add_subplot(122)
contour = ax2.contour(A0, A1, J_outlier, levels=50, cmap="viridis")
ax2.set_xlabel("$\\theta_0$")
ax2.set_ylabel("$\\theta_1$")
ax2.set_title("Curvas de nivel de la función de coste con un valor atípico")
plt.colorbar(contour, ax=ax2)
plt.tight_layout()
plt.show()

global_min_index_outlier = np.unravel_index(np.argmin(J_outlier), J_outlier.shape)
optimal_a0_outlier = a0_list[global_min_index_outlier[1]]
optimal_a1_outlier = a1_list[global_min_index_outlier[0]]
print(f"Valor óptimo de theta_0 con un valor atípico: {optimal_a0_outlier}")
print(f"Valor óptimo de theta_1 con un valor atípico: {optimal_a1_outlier}")

display(
    Markdown(
        "Al introducir un valor atípico en el conjunto de datos, la función de coste se ve significativamente afectada, lo que resulta en un mínimo global que se encuentra en un punto muy diferente al caso ideal. La función de coste cuadrática es sensible a los valores atípicos, ya que estos pueden aumentar drásticamente el error cuadrático, desplazando el mínimo global hacia parámetros que no representan adecuadamente la tendencia general de los datos. Esto destaca una limitación importante de la función de coste cuadrática: su sensibilidad a los valores atípicos, lo que puede llevar a modelos que no generalizan bien a nuevos datos."
    )
)

# 8. Compare el ajuste obtenido con un modelo lineal sobre un conjunto de datos no lineales. Discuta si minimizar la función de coste garantiza que el modelo sea adecuado.
# %%

non_linear_df = pd.DataFrame({"X_1": np.arange(6), "Y": np.array([0, 1, 4, 9, 16, 25])})
non_linear_df


# %%
J_non_linear = np.array(
    [
        [func_coste(a0, a1, non_linear_df["X_1"], non_linear_df["Y"]) for a0 in a0_list]
        for a1 in a1_list
    ]
)
global_min_index_non_linear = np.unravel_index(
    np.argmin(J_non_linear), J_non_linear.shape
)
optimal_a0_non_linear = a0_list[global_min_index_non_linear[1]]
optimal_a1_non_linear = a1_list[global_min_index_non_linear[0]]
print(f"Valor óptimo de theta_0 para datos no lineales: {optimal_a0_non_linear}")
print(f"Valor óptimo de theta_1 para datos no lineales: {optimal_a1_non_linear}")

# %%

# Graficar el ajuste del modelo lineal sobre los datos no lineales
plt.plot(non_linear_df["X_1"], non_linear_df["Y"], "o", label="Datos no lineales")
plt.plot(
    non_linear_df["X_1"],
    h(optimal_a0_non_linear, optimal_a1_non_linear, non_linear_df["X_1"]),
    label="Modelo lineal ajustado",
    color="red",
)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Ajuste de un modelo lineal sobre datos no lineales")
plt.legend()
plt.show()

display(
    Markdown(
        "Minimizar la función de coste para un modelo lineal sobre un conjunto de datos no lineales puede resultar en un ajuste que no capture adecuadamente la relación entre las variables. En este caso, aunque se encuentra un mínimo global para los parámetros del modelo lineal, el ajuste resultante no representa bien la tendencia cuadrática de los datos. Esto demuestra que minimizar la función de coste no garantiza que el modelo sea adecuado. Es importante considerar la naturaleza de los datos y elegir un modelo que pueda capturar esa relación de manera efectiva."
    )
)

# %% [markdown]
# # 9. Obtenga la expresión teórica de la función de coste en el caso con un parámetro y con dos parámetros, e interprete el significado de sus mínimos.
# %% [markdown]
# * Para el caso de la regresion lineal 1D, partimos de la expresion:

# \begin{equation}
# J(\theta_1, \theta_2)=\frac{1}{2m} \sum_{i=0}^m ( h_{\theta} (x^{(i)})-y^{(i)})^2
# \end{equation}

# Y de la hipotesis

# \begin{equation}
# h(X) = \theta_0 + \theta_1 X
# \end{equation}

# Con lo cual obtenemos:

# \begin{equation}
# J(\theta_1, \theta_2)=\frac{1}{2m} \sum_{i=0}^m ( (\theta_0 + \theta_1 x^{(i)})- y^{(i)})^2
# \end{equation}

# * Para el caso 2D, partimos de la hipotesis:

# \begin{equation}
# h(X) = \theta_0 + \theta_1 X_1 + \theta_2 X_2
# \end{equation}

# Con lo cual obtenemos:

# \begin{equation}
# J(\theta_1, \theta_2)=\frac{1}{2m} \sum_{i=0}^m ( (\theta_0 + \theta_1 x_1^{(i)} + \theta_2 x_2^{(i)})- y^{(i)})^2
# \end{equation}

# %% [markdown]
# # 10. Construya un algoritmo en el que emplee el gradiente descendente para determinar el mínimo de una función. Determine dicho mínimo con un error $\epsilon$ de $10^{-4}$. Pruebe su algoritmo para

# $$
# f(x)=(x-4)^2
# $$

# y al menos tres valores diferentes de $\alpha$.

# %%


def gradient_descent(alpha, epsilon=1e-4):
    x = 0  # Valor inicial
    while True:
        grad = 2 * (x - 4)  # Gradiente de f(x)
        x_new = x - alpha * grad  # Actualización de x
        if abs(f(x_new) - f(x)) < epsilon:  # Verificar la condición de convergencia
            break
        x = x_new
    return x


# %%
def f(x):
    return (x - 4) ** 2


# %%
alphas = [0.01, 0.1, 0.5]
# %%
for alpha in alphas:
    minimum = gradient_descent(alpha)
    print(f"Valor mínimo de f(x) con alpha={alpha}: {f(minimum)} en x={minimum}")
# %% [markdown]
# # 11. Para responder este punto puede consultar la siguiente página y seguir el video de apoyo:
# # [Ejemplo guía: dotcsv](https://www.youtube.com/watch?v=-_A_AAxqzCg)
#
# Encontrar el mínimo de la siguiente función a través del método del gradiente descendente:
#
# $$
# F(x,y)=\sin\left(\frac{1}{2}x^2-\frac{1}{4}y^2+3\right)\cos\left(2x+1-e^y\right)
# $$
#
# - Para ello, realice una gráfica de la función en 3D y un mapa de contorno de la función.
# - Determine el valor mínimo de la función con el método del gradiente descendente.


# %%
def F(x, y):
    return np.sin(0.5 * x**2 - 0.25 * y**2 + 3) * np.cos(2 * x + 1 - np.exp(y))


# %%
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)
# %%
# Gráfica en 3D
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title="Gráfica en 3D de F(x,y)", autosize=True)
fig.show()
# %%
# Mapa de contorno
fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale="Viridis"))
fig.update_layout(title="Mapa de contorno de F(x,y)", autosize=True)
fig.show()


# %%
# Gradiente descendente para encontrar el mínimo de F(x,y)
def gradient_descent_2d(alpha, epsilon=1e-4):
    x, y = np.random.randint(-10, 10), np.random.randint(-10, 10)  # Valores iniciales
    while True:
        grad_x = 0.5 * x * np.cos(0.5 * x**2 - 0.25 * y**2 + 3) * np.cos(
            2 * x + 1 - np.exp(y)
        ) - 2 * np.sin(0.5 * x**2 - 0.25 * y**2 + 3) * np.sin(2 * x + 1 - np.exp(y))
        grad_y = -0.25 * y * np.cos(0.5 * x**2 - 0.25 * y**2 + 3) * np.cos(
            2 * x + 1 - np.exp(y)
        ) + np.sin(0.5 * x**2 - 0.25 * y**2 + 3) * np.sin(
            2 * x + 1 - np.exp(y)
        ) * np.exp(y)
        x_new = x - alpha * grad_x
        y_new = y - alpha * grad_y
        if abs(F(x_new, y_new) - F(x, y)) < epsilon:
            break
        x, y = x_new, y_new
    return x, y


# %%
alpha = 0.01
minimum_x, minimum_y = gradient_descent_2d(alpha)
display(
    Markdown(
        f"Valor mínimo de F(x,y) con alpha={alpha}: {F(minimum_x, minimum_y)} en (x, y)=({minimum_x}, {minimum_y})"
    )
)

# %% [markdown]
# # 12. Empleando los siguientes datos:

# ```python
# X = np.linspace(0, 1, 100)
# y = 0.2 + 0.2*X + 0.02*np.random.random(100)
# ```


# y las herramientas desarrolladas en los apartados anteriores, construya un algorítmo que permita determinar una regresión lineal.
# %%
def linear_regression(X, y, theta0, theta1, alpha, max_iter=10000):
    m = len(X)
    Js = [func_coste(theta0, theta1, X, y)]
    for i in range(max_iter):
        grad0 = np.sum((theta0 + theta1 * X - y)) / m
        grad1 = np.sum((theta0 + theta1 * X - y) * X) / m
        theta0 -= alpha * grad0
        theta1 -= alpha * grad1
        J = func_coste(theta0, theta1, X, y)
        Js.append(J)
        # Compute the gradient
        if abs(Js[-1] - Js[-2]) < 1e-12:
            return theta0, theta1, i, Js

    return theta0, theta1, max_iter, Js


# %%
X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)
theta0, theta1, iterations, Js = linear_regression(X, y, theta0=0, theta1=0, alpha=0.1)
print(f"theta0: {theta0}, theta1: {theta1}, iterations: {iterations}")
plt.plot(X, y, "o", label="Datos")
plt.plot(X, theta0 + theta1 * X, label="Regresión lineal", color="red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión lineal ajustada a los datos")
plt.legend()
plt.show()

# %% [markdown]
# # 13. Compare su resultado empleando la libreria linearRegresion() de sklearn.

# %%
# Regrsion lineal con sklearn

L = LinearRegression()

L.fit(X.reshape(-1, 1), y)
theta0 = L.intercept_
theta1 = L.coef_[0]
print(f"theta0: {theta0}, theta1: {theta1}")
plt.plot(X, y, "o", label="Datos")
plt.plot(X, theta0 + theta1 * X, label="Regresión lineal (sklearn)", color="green")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión lineal ajustada a los datos (sklearn)")
plt.legend()
plt.show()
# %% [markdown]
# El resultado obtenido con el algoritmo de gradiente descendente es similar al obtenido con la función `LinearRegression()` de sklearn, lo que indica que el algoritmo implementado es correcto y capaz de encontrar los parámetros de la regresión lineal que mejor se ajustan a los datos.
