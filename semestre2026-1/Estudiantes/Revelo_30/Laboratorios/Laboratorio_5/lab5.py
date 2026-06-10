import matplotlib.pyplot as plt
import numpy as np


# Punto 1
def y(x1, x2):
    return 2.1 * x1 + 3.1 * x2


x1 = np.random.randint(-10, 10, 100)
x2 = np.random.randint(-10, 10, 100)

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x1, x2, y(x1, x2))
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
plt.title("Puntos aleatorios sobre el plano")
plt.show()


# Punto 2
theta = np.random.randn(3, 1) * np.random.randint(1, 5)

# Punto 3
ones = np.ones(100)
X = np.array([ones, x1, x2])
Y = np.array(y(x1, x2)).reshape(1, 100)

print("X.shape =", X.shape)
print("Y.shape =", Y.shape)


# Punto 4
h = np.dot(theta.T, X)
L = h - Y
L2 = L**2

m = len(x1)
J = (1 / (2 * m)) * np.sum(L2)
print("Costo inicial:", J)

# Punto 5
grad_J = (1 / m) * np.dot(X, L.T)
theta = theta - 0.01 * grad_J

# Punto 6
alpha = 0.01
num_iters = 10000

for i in range(num_iters):
    h = np.dot(theta.T, X)
    L = h - Y
    grad_J = np.dot(X, L.T) / m
    theta = theta - alpha * grad_J

print("Theta final:")
print(theta.round(2))
h = np.dot(theta.T, X)
L = h - Y
J_final = (1 / (2 * m)) * np.sum(L**2)

print("Costo final:", J_final)


# Punto 7
class MultilinearRegresion:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        Nfeatures = X.shape[0]
        self.m = X.shape[1]
        self.theta = np.random.random(Nfeatures).reshape(Nfeatures, 1)

    def model(self):
        self.h = np.dot(self.theta.T, self.X)

    def costo(self):
        L = self.h - self.Y
        self.J = (1 / (2 * self.m)) * np.sum(L**2)

    def update_params(self, learning_rate):
        grad = (1 / self.m) * np.dot(self.X, (self.h - self.Y).T)
        self.theta = self.theta - learning_rate * grad

    def fit(self, learning_rate, num_iters=10000):
        for i in range(num_iters):
            self.model()
            self.costo()
            self.update_params(learning_rate)


modelo = MultilinearRegresion(X, Y)
modelo.fit(0.01, 10000)

print("Theta final de la clase:")
print(modelo.theta.round(2))
print("Costo final de la clase:")
print(modelo.J)
