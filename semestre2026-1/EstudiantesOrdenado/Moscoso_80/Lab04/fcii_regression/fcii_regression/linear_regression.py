"""
Implementación de Regresión Lineal usando Gradiente Descendente.

Este módulo contiene la clase LinearRegression que implementa un modelo
de regresión lineal simple (y = θ₀ + θ₁x) optimizado mediante gradiente
descendente con función de coste MSE.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from .gradient_descent import GradientDescent
from .cost_functions import mean_squared_error


class LinearRegression:
    """
    Modelo de Regresión Lineal Simple optimizado con Gradiente Descendente.

    Implementa un modelo de la forma:
        h(x) = θ₀ + θ₁ * x

    donde θ₀ es el intercepto (bias) y θ₁ es la pendiente.

    El modelo se entrena minimizando la función de coste MSE:
        J(θ) = (1/2m) * Σ(h(xᵢ) - yᵢ)²

    Attributes:
        optimizer (GradientDescent): Instancia del optimizador a usar.
        theta_ (np.ndarray): Parámetros entrenados [θ₀, θ₁] (disponible tras fit).
        cost_history_ (np.ndarray): Historial de valores de coste durante entrenamiento.

    Example:
        >>> from fcii_regression import GradientDescent, LinearRegression
        >>> import numpy as np
        >>>
        >>> # Crear datos
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> y = 2 * X + 1 + np.random.randn(5) * 0.5
        >>>
        >>> # Configurar optimizador
        >>> gd = GradientDescent(n_iterations=1000, learning_rate=0.01)
        >>>
        >>> # Entrenar modelo
        >>> model = LinearRegression(optimizer=gd)
        >>> model.fit(X, y)
        >>>
        >>> # Predecir
        >>> y_pred = model.predict(X)
        >>> print(f"θ₀ = {model.theta_[0]:.4f}, θ₁ = {model.theta_[1]:.4f}")
    """

    def __init__(self, optimizer: Optional[GradientDescent] = None):
        """
        Inicializa el modelo de Regresión Lineal.

        Args:
            optimizer (GradientDescent, optional): Instancia del optimizador
                de gradiente descendente. Si es None, se crea uno con
                parámetros por defecto.
        """
        if optimizer is None:
            self.optimizer = GradientDescent()
        else:
            self.optimizer = optimizer

        # Atributos calculados tras fit()
        self.theta_: Optional[np.ndarray] = None
        self.cost_history_: Optional[np.ndarray] = None
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    def _hypothesis(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Calcula la hipótesis lineal h(x) = θ₀ + θ₁ * x.

        En forma vectorizada: h(X) = X @ θ, donde X tiene una columna de 1s.

        Args:
            X (np.ndarray): Features con columna de 1s. Shape: (m, 2).
            theta (np.ndarray): Parámetros [θ₀, θ₁]. Shape: (2,).

        Returns:
            np.ndarray: Predicciones h(x) para cada muestra. Shape: (m,).
        """
        return X.dot(theta)

    def _add_bias_column(self, X: np.ndarray) -> np.ndarray:
        """
        Agrega una columna de unos para el término de bias (θ₀).

        Transforma X de shape (m,) a (m, 2) agregando una columna de 1s.

        Args:
            X (np.ndarray): Features originales. Shape: (m,) o (m, 1).

        Returns:
            np.ndarray: Features con columna de bias. Shape: (m, 2).
        """
        X = np.asarray(X).flatten()
        return np.column_stack([np.ones(len(X)), X])

    def _cost_function(self, theta: np.ndarray) -> float:
        """
        Calcula el coste J(θ) = MSE para los datos de entrenamiento.

        Esta función es usada internamente por el optimizador.

        Args:
            theta (np.ndarray): Parámetros [θ₀, θ₁].

        Returns:
            float: Valor del coste MSE.
        """
        y_pred = self._hypothesis(self._X_train, theta)
        return mean_squared_error(self._y_train, y_pred)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_theta: Optional[np.ndarray] = None
    ) -> 'LinearRegression':
        """
        Entrena el modelo de regresión lineal.

        Usa el optimizador de gradiente descendente para encontrar los
        parámetros θ que minimizan la función de coste MSE.

        Args:
            X (np.ndarray): Features de entrenamiento. Shape: (m,).
            y (np.ndarray): Valores objetivo. Shape: (m,).
            initial_theta (np.ndarray, optional): Parámetros iniciales [θ₀, θ₁].
                Si es None, se inicializan aleatoriamente.

        Returns:
            LinearRegression: self, para encadenar métodos.
        """
        # Preparar datos
        self._X_train = self._add_bias_column(X)
        self._y_train = np.asarray(y).flatten()

        # Punto inicial por defecto
        if initial_theta is None:
            initial_theta = np.random.randn(2) * 0.5

        # Ejecutar optimización
        result = self.optimizer.minimize(
            func=self._cost_function,
            initial_point=initial_theta,
            n_params=2
        )

        # Guardar resultados
        self.theta_ = result['minimum_point']
        self.cost_history_ = result['values']

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones usando el modelo entrenado.

        Args:
            X (np.ndarray): Features para predecir. Shape: (m,).

        Returns:
            np.ndarray: Predicciones ŷ = θ₀ + θ₁ * x. Shape: (m,).

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if self.theta_ is None:
            raise ValueError("El modelo debe ser entrenado con fit() primero.")

        X_with_bias = self._add_bias_column(X)
        return self._hypothesis(X_with_bias, self.theta_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el coeficiente de determinación R² del modelo.

        R² = 1 - SS_res / SS_tot

        donde:
            SS_res = Σ(yᵢ - ŷᵢ)² (suma de cuadrados residuales)
            SS_tot = Σ(yᵢ - ȳ)² (suma de cuadrados totales)

        Args:
            X (np.ndarray): Features. Shape: (m,).
            y (np.ndarray): Valores reales. Shape: (m,).

        Returns:
            float: Coeficiente R² (entre 0 y 1, mayor es mejor).
        """
        y = np.asarray(y).flatten()
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def get_equation(self) -> str:
        """
        Devuelve la ecuación del modelo en formato legible.

        Returns:
            str: Ecuación "y = θ₀ + θ₁ * x" con valores numéricos.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if self.theta_ is None:
            raise ValueError("El modelo debe ser entrenado con fit() primero.")

        sign = "+" if self.theta_[1] >= 0 else "-"
        return f"y = {self.theta_[0]:.4f} {sign} {abs(self.theta_[1]):.4f} * x"

    def plot_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        title: str = "Regresión Lineal - Ajuste del Modelo"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Grafica los datos y la línea de regresión ajustada.

        Args:
            X (np.ndarray): Features. Shape: (m,).
            y (np.ndarray): Valores reales. Shape: (m,).
            title (str): Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Datos originales
        ax.scatter(X, y, color='blue', alpha=0.6, label='Datos', edgecolors='black')

        # Línea de regresión
        X_line = np.linspace(X.min(), X.max(), 100)
        y_line = self.predict(X_line)
        ax.plot(X_line, y_line, color='red', linewidth=2, label=f'Ajuste: {self.get_equation()}')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig, ax

    def plot_cost_history(
        self,
        title: str = "Evolución de la Función de Coste"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Grafica la evolución del coste J(θ) durante el entrenamiento.

        Args:
            title (str): Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.

        Raises:
            ValueError: Si el modelo no ha sido entrenado.
        """
        if self.cost_history_ is None:
            raise ValueError("El modelo debe ser entrenado con fit() primero.")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(self.cost_history_, color='blue', linewidth=1)
        ax.set_xlabel('Iteración')
        ax.set_ylabel('J(θ) - MSE')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        final_cost = self.cost_history_[-1]
        ax.axhline(
            y=final_cost, color='red', linestyle='--',
            label=f'Coste final = {final_cost:.6f}'
        )
        ax.legend()

        return fig, ax

    def plot_cost_surface(
        self,
        theta0_range: Tuple[float, float] = (-5, 5),
        theta1_range: Tuple[float, float] = (-5, 5),
        resolution: int = 100,
        title: str = "Superficie de la Función de Coste"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Grafica la superficie 3D de la función de coste J(θ₀, θ₁).

        Args:
            theta0_range: Rango para θ₀. Default: (-5, 5).
            theta1_range: Rango para θ₁. Default: (-5, 5).
            resolution: Número de puntos por eje. Default: 100.
            title: Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.
        """
        return self.optimizer.plot_3d_surface(
            func=self._cost_function,
            x_range=theta0_range,
            y_range=theta1_range,
            resolution=resolution,
            show_trajectory=True,
            title=title
        )

    def plot_cost_contour(
        self,
        theta0_range: Tuple[float, float] = (-5, 5),
        theta1_range: Tuple[float, float] = (-5, 5),
        resolution: int = 100,
        title: str = "Mapa de Contorno de la Función de Coste"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Grafica el mapa de contorno de la función de coste J(θ₀, θ₁).

        Args:
            theta0_range: Rango para θ₀. Default: (-5, 5).
            theta1_range: Rango para θ₁. Default: (-5, 5).
            resolution: Número de puntos por eje. Default: 100.
            title: Título del gráfico.

        Returns:
            tuple: (fig, ax) objetos de matplotlib.
        """
        return self.optimizer.plot_contour(
            func=self._cost_function,
            x_range=theta0_range,
            y_range=theta1_range,
            resolution=resolution,
            show_trajectory=True,
            title=title
        )
