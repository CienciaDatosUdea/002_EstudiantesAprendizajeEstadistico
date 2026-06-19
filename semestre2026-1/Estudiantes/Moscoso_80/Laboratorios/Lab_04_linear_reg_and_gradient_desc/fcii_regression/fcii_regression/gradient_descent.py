"""
Implementación del algoritmo de Gradiente Descendente.

Este módulo contiene la clase GradientDescent que implementa el algoritmo
de optimización iterativa para encontrar mínimos de funciones.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Optional, Tuple, Dict, Any


class GradientDescent:
    """
    Optimizador de Gradiente Descendente para minimización de funciones.

    El gradiente descendente es un algoritmo de optimización iterativo que
    busca el mínimo de una función moviéndose en la dirección opuesta al
    gradiente. La regla de actualización es:

        θ_new = θ_old - α * ∇F(θ)

    donde α es el learning rate (tasa de aprendizaje).

    Attributes:
        n_iterations (int): Número máximo de iteraciones.
        learning_rate (float): Tasa de aprendizaje (α).
        h (float): Paso para aproximación numérica del gradiente.
        verbose (bool): Si True, imprime progreso durante la optimización.
        history_ (np.ndarray): Historial de puntos visitados (disponible tras fit).
        values_ (np.ndarray): Valores de la función en cada iteración.

    Example:
        >>> gd = GradientDescent(n_iterations=1000, learning_rate=0.01)
        >>> func = lambda x: x[0]**2 + x[1]**2
        >>> result = gd.minimize(func, initial_point=[5, 5])
        >>> print(f"Mínimo en: {result['minimum_point']}")
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        h: float = 0.001,
        verbose: bool = False
    ):
        """
        Inicializa el optimizador de Gradiente Descendente.

        Args:
            n_iterations (int): Número máximo de iteraciones del algoritmo.
                               Default: 1000.
            learning_rate (float): Tasa de aprendizaje (α). Controla el tamaño
                                   del paso en cada iteración. Valores muy
                                   grandes causan divergencia, muy pequeños
                                   ralentizan la convergencia. Default: 0.01.
            h (float): Paso para la aproximación numérica del gradiente usando
                       diferencias finitas. Default: 0.001.
            verbose (bool): Si True, imprime el valor de la función cada 100
                           iteraciones. Default: False.
        """
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.h = h
        self.verbose = verbose

        # Atributos calculados tras minimize()
        self.history_: Optional[np.ndarray] = None
        self.values_: Optional[np.ndarray] = None
        self.minimum_point_: Optional[np.ndarray] = None
        self.minimum_value_: Optional[float] = None

    def _compute_numerical_gradient(
        self,
        func: Callable[[np.ndarray], float],
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Calcula el gradiente numéricamente usando diferencias finitas.

        Aproxima cada derivada parcial usando:
            ∂F/∂θᵢ ≈ (F(θ + h*eᵢ) - F(θ)) / h

        donde eᵢ es el vector unitario en la dirección i.

        Args:
            func: Función a derivar.
            theta: Punto donde calcular el gradiente.

        Returns:
            np.ndarray: Vector gradiente.
        """
        grad = np.zeros_like(theta, dtype=float)
        for i in range(len(theta)):
            theta_plus_h = theta.copy()
            theta_plus_h[i] += self.h
            grad[i] = (func(theta_plus_h) - func(theta)) / self.h
        return grad

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        initial_point: Optional[np.ndarray] = None,
        n_params: int = 2
    ) -> Dict[str, Any]:
        """
        Encuentra el mínimo local de una función usando gradiente descendente.

        El algoritmo aproxima el gradiente usando diferencias finitas
        (forward differences) y actualiza los parámetros iterativamente.

        Args:
            func (callable): Función a minimizar. Debe recibir un array de
                            parámetros y devolver un escalar.
            initial_point (array-like or None): Punto inicial. Si es None,
                                                se genera aleatoriamente en [-2, 2].
            n_params (int): Número de parámetros (solo usado si initial_point=None).

        Returns:
            dict: Diccionario con:
                - 'minimum_point': array con el punto mínimo encontrado.
                - 'minimum_value': valor de la función en el mínimo.
                - 'history': array de puntos visitados (shape: n_iter+1, n_params).
                - 'values': array de valores de la función en cada iteración.
        """
        # Inicializar punto de partida
        if initial_point is None:
            theta = np.random.rand(n_params) * 4 - 2
        else:
            theta = np.array(initial_point, dtype=float)

        # Almacenar historial
        history = [theta.copy()]
        values = [func(theta)]

        for iteration in range(self.n_iterations):
            # Calcular gradiente numérico
            grad = self._compute_numerical_gradient(func, theta)

            # Actualizar parámetros: θ = θ - α * ∇F
            theta = theta - self.learning_rate * grad

            # Guardar historial
            history.append(theta.copy())
            values.append(func(theta))

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteración {iteration + 1}: F = {func(theta):.6f}")

        # Guardar resultados como atributos
        self.history_ = np.array(history)
        self.values_ = np.array(values)
        self.minimum_point_ = theta
        self.minimum_value_ = func(theta)

        return {
            'minimum_point': self.minimum_point_,
            'minimum_value': self.minimum_value_,
            'history': self.history_,
            'values': self.values_
        }

    def plot_3d_surface(
        self,
        func: Callable[[np.ndarray], float],
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        resolution: int = 100,
        show_trajectory: bool = True,
        title: str = "Superficie 3D de la Función de Coste"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera una visualización 3D de la función con la trayectoria del GD.

        Args:
            func: Función a graficar. Debe recibir [x, y] y devolver escalar.
            x_range: Rango (min, max) para el eje x. Default: (-3, 3).
            y_range: Rango (min, max) para el eje y. Default: (-3, 3).
            resolution: Número de puntos en cada eje. Default: 100.
            show_trajectory: Si True, muestra la trayectoria del GD. Default: True.
            title: Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Evaluar la función en la malla
        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, label='F(θ₀, θ₁)')

        # Superponer trayectoria si existe y se solicita
        if show_trajectory and self.history_ is not None:
            z_traj = np.array([func(p) for p in self.history_])
            ax.plot(
                self.history_[:, 0], self.history_[:, 1], z_traj,
                color='red', linewidth=2, marker='o', markersize=2,
                label='Trayectoria GD'
            )
            ax.scatter(
                *self.history_[-1], z_traj[-1],
                color='yellow', s=100, marker='*',
                label='Mínimo encontrado', zorder=5
            )
            ax.legend()

        ax.set_xlabel('θ₀')
        ax.set_ylabel('θ₁')
        ax.set_zlabel('J(θ)')
        ax.set_title(title)

        return fig, ax

    def plot_contour(
        self,
        func: Callable[[np.ndarray], float],
        x_range: Tuple[float, float] = (-3, 3),
        y_range: Tuple[float, float] = (-3, 3),
        resolution: int = 100,
        show_trajectory: bool = True,
        title: str = "Mapa de Contorno de la Función de Coste"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera un mapa de contorno (curvas de nivel) de la función.

        Args:
            func: Función a graficar. Debe recibir [x, y] y devolver escalar.
            x_range: Rango (min, max) para el eje x. Default: (-3, 3).
            y_range: Rango (min, max) para el eje y. Default: (-3, 3).
            resolution: Número de puntos en cada eje. Default: 100.
            show_trajectory: Si True, muestra la trayectoria del GD. Default: True.
            title: Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(resolution):
            for j in range(resolution):
                Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

        fig, ax = plt.subplots(figsize=(10, 8))

        contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
        ax.contour(X, Y, Z, levels=20, colors='white', alpha=0.3, linewidths=0.5)
        plt.colorbar(contour, ax=ax, label='J(θ)')

        # Superponer trayectoria si existe y se solicita
        if show_trajectory and self.history_ is not None:
            ax.plot(
                self.history_[:, 0], self.history_[:, 1],
                color='red', linewidth=1.5, marker='.', markersize=3,
                label='Trayectoria GD', alpha=0.8
            )
            ax.scatter(
                *self.history_[0], color='green', s=100, marker='o',
                label='Punto inicial', zorder=5, edgecolors='white'
            )
            ax.scatter(
                *self.history_[-1], color='yellow', s=150, marker='*',
                label='Mínimo encontrado', zorder=5, edgecolors='black'
            )
            ax.legend(loc='upper right')

        ax.set_xlabel('θ₀')
        ax.set_ylabel('θ₁')
        ax.set_title(title)

        return fig, ax

    def plot_convergence(
        self,
        title: str = "Convergencia del Gradiente Descendente"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Genera una gráfica de convergencia mostrando J(θ) vs iteraciones.

        Args:
            title: Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.

        Raises:
            ValueError: Si no se ha ejecutado minimize() previamente.
        """
        if self.values_ is None:
            raise ValueError("Primero debe ejecutar minimize() antes de graficar.")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.values_, color='blue', linewidth=1)
        ax.set_xlabel('Iteración')
        ax.set_ylabel('J(θ)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(
            y=self.minimum_value_, color='red', linestyle='--',
            label=f'Mínimo = {self.minimum_value_:.4f}'
        )
        ax.legend()

        return fig, ax
