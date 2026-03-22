"""
Librería de Regresión Lineal con Gradiente Descendente

Este módulo implementa un modelo de regresión lineal simple usando
la técnica de gradiente descendente para optimizar los parámetros.

Autor: Estudiante de Física Computacional II
Universidad de Antioquia - 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


class LinearRegression:
    """
    Modelo de regresión lineal simple mediante gradiente descendente.
    
    Este modelo implementa regresión lineal univariada (una variable independiente)
    de la forma: y = θ₀ + θ₁*x
    
    Attributes:
        theta_0 (float): Parámetro de intercepto (pendiente y)
        theta_1 (float): Parámetro de pendiente
        cost_history (list): Historial de valores de coste durante el entrenamiento
        params_history (list): Historial de parámetros durante el entrenamiento
    """
    
    def __init__(self):
        """Inicializa el modelo con parámetros en cero."""
        self.theta_0 = 0.0
        self.theta_1 = 0.0
        self.cost_history = []
        self.params_history = []
    
    def hypothesis(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula la hipótesis lineal h(X) = θ₀ + θ₁*X
        
        Una hipótesis (o modelo) es la función que genera predicciones.
        Para regresión lineal simple, es una función lineal de X.
        
        Args:
            X (np.ndarray): Array 1D con valores de las variables independientes
        
        Returns:
            np.ndarray: Array con las predicciones del modelo
        
        Raises:
            ValueError: Si X no es un array de numpy
        
        Example:
            >>> model = LinearRegression()
            >>> model.theta_0, model.theta_1 = 2.0, 3.0
            >>> X = np.array([1, 2, 3])
            >>> predictions = model.hypothesis(X)
            >>> # predictions = [5, 8, 11]
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X debe ser un array de numpy")
        
        return self.theta_0 + self.theta_1 * X
    
    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula la función de coste (error cuadrado medio).
        
        La función de coste mide qué tan bien el modelo se ajusta a los datos.
        Usamos el error cuadrado medio (MSE):
        
        J(θ₀, θ₁) = (1/2m) * Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        
        donde m es el número de muestras.
        
        Args:
            X (np.ndarray): Array 1D con variables independientes
            y (np.ndarray): Array 1D con variables dependientes (observadas)
        
        Returns:
            float: Valor de la función de coste
        
        Raises:
            ValueError: Si las dimensiones de X e y no coinciden
        
        Example:
            >>> model = LinearRegression()
            >>> model.theta_0, model.theta_1 = 0.0, 2.0
            >>> X = np.array([1.0, 2.0, 3.0])
            >>> y = np.array([2.0, 4.0, 6.0])
            >>> cost = model.cost_function(X, y)
            >>> # cost = 0.0 (ajuste perfecto)
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y deben tener el mismo número de muestras")
        
        m = len(X)
        h = self.hypothesis(X)  # predicciones
        error = h - y  # residuales (errores)
        J = (1 / (2 * m)) * np.sum(error ** 2)
        
        return J
    
    def gradient_descent(self, X: np.ndarray, y: np.ndarray,
                        learning_rate: float = 0.01,
                        epsilon: float = 1e-6,
                        max_iterations: int = 10000,
                        verbose: bool = False) -> Dict:
        """
        Ejecuta el algoritmo de gradiente descendente para minimizar J(θ).
        
        El gradiente descendente es un algoritmo de optimización iterativo que
        actualiza los parámetros en la dirección del descenso más pronunciado:
        
        θ₀ := θ₀ - α * ∂J/∂θ₀
        θ₁ := θ₁ - α * ∂J/∂θ₁
        
        donde α es la tasa de aprendizaje (learning_rate).
        
        Args:
            X (np.ndarray): Array 1D con variables independientes
            y (np.ndarray): Array 1D con variables dependientes (objetivos)
            learning_rate (float): Tasa de aprendizaje α. Valor por defecto: 0.01
            epsilon (float): Criterio de convergencia. Por defecto: 1e-6
            max_iterations (int): Número máximo de iteraciones. Por defecto: 10000
            verbose (bool): Si True, imprime información durante el entrenamiento
        
        Returns:
            dict: Diccionario con información del entrenamiento:
                - 'theta_0': Parámetro θ₀ optimizado
                - 'theta_1': Parámetro θ₁ optimizado
                - 'cost': Valor final de coste
                - 'iterations': Número de iteraciones realizadas
                - 'converged': True si convergió antes de max_iterations
        
        Raises:
            ValueError: Si learning_rate, epsilon o max_iterations no son válidos
        
        Example:
            >>> model = LinearRegression()
            >>> X = np.array([1, 2, 3, 4, 5], dtype=float)
            >>> y = np.array([2, 4, 5, 4, 5], dtype=float)
            >>> result = model.gradient_descent(X, y, learning_rate=0.01, verbose=True)
            >>> print(f"θ₀ = {result['theta_0']:.4f}, θ₁ = {result['theta_1']:.4f}")
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if epsilon <= 0:
            raise ValueError("epsilon debe ser positivo")
        if max_iterations <= 0:
            raise ValueError("max_iterations debe ser positivo")
        
        m = len(X)
        self.cost_history = []
        self.params_history = []
        converged = False
        
        for iteration in range(max_iterations):
            # Calcular predicciones
            h = self.hypothesis(X)
            error = h - y
            
            # Calcular coste actual
            J = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(J)
            self.params_history.append((self.theta_0, self.theta_1))
            
            # Calcular gradientes parciales
            grad_theta_0 = (1 / m) * np.sum(error)
            grad_theta_1 = (1 / m) * np.sum(error * X)
            
            # Guardar parámetros anteriores para comprobar convergencia
            theta_0_old = self.theta_0
            theta_1_old = self.theta_1
            
            # Actualizar parámetros
            self.theta_0 = self.theta_0 - learning_rate * grad_theta_0
            self.theta_1 = self.theta_1 - learning_rate * grad_theta_1
            
            # Imprimir progreso si verbose=True
            if verbose and (iteration % max(1, max_iterations // 10) == 0):
                print(f"Iteración {iteration}: J = {J:.6f}, "
                      f"θ₀ = {self.theta_0:.6f}, θ₁ = {self.theta_1:.6f}")
            
            # Verificar convergencia
            if iteration > 0:
                param_change = np.sqrt((self.theta_0 - theta_0_old)**2 + 
                                      (self.theta_1 - theta_1_old)**2)
                if param_change < epsilon:
                    converged = True
                    if verbose:
                        print(f"¡Convergencia alcanzada en iteración {iteration + 1}!")
                    break
        
        J_final = self.cost_function(X, y)
        
        return {
            'theta_0': self.theta_0,
            'theta_1': self.theta_1,
            'cost': J_final,
            'iterations': iteration + 1,
            'converged': converged
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            learning_rate: float = 0.01,
            epsilon: float = 1e-6,
            max_iterations: int = 10000,
            verbose: bool = False) -> 'LinearRegression':
        """
        Método principal para ajustar el modelo a los datos.
        
        Este método entrena el modelo usando el algoritmo de gradiente descendente.
        Es el método que debe usar el usuario para ajustar el modelo.
        
        Args:
            X (np.ndarray): Array 1D con variables independientes
            y (np.ndarray): Array 1D con variables dependientes (observadas)
            learning_rate (float): Tasa de aprendizaje. Por defecto: 0.01
            epsilon (float): Criterio de convergencia. Por defecto: 1e-6
            max_iterations (int): Número máximo de iteraciones. Por defecto: 10000
            verbose (bool): Si True, imprime información de entrenamiento
        
        Returns:
            LinearRegression: El modelo entrenado (self)
        
        Example:
            >>> model = LinearRegression()
            >>> X = np.array([1, 2, 3, 4, 5], dtype=float)
            >>> y = np.array([2, 4, 5, 4, 5], dtype=float)
            >>> model.fit(X, y, learning_rate=0.01)
            >>> predictions = model.predict(X)
        """
        self.gradient_descent(X, y, learning_rate, epsilon, max_iterations, verbose)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones sobre nuevos datos.
        
        Args:
            X (np.ndarray): Array 1D con variables independientes
        
        Returns:
            np.ndarray: Array con las predicciones
        
        Example:
            >>> model = LinearRegression()
            >>> model.fit(X_train, y_train)
            >>> y_pred = model.predict(X_test)
        """
        return self.hypothesis(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula el coeficiente de determinación R².
        
        R² mide qué proporción de la varianza en y es explicada por el modelo.
        R² = 1 - (SS_res / SS_tot)
        
        donde SS_res = Σ(y - ŷ)² y SS_tot = Σ(y - ȳ)²
        
        Args:
            X (np.ndarray): Array con variables independientes
            y (np.ndarray): Array con variables dependientes observadas
        
        Returns:
            float: Coeficiente R² (entre 0 y 1, donde 1 es ajuste perfecto)
        
        Example:
            >>> model = LinearRegression()
            >>> model.fit(X_train, y_train)
            >>> r2 = model.score(X_test, y_test)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_results(self, X: np.ndarray, y: np.ndarray, title: str = None):
        """
        Genera gráficas del ajuste del modelo y la convergencia.
        
        Args:
            X (np.ndarray): Array con variables independientes
            y (np.ndarray): Array con variables dependientes
            title (str): Título opcional para la gráfica
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfica 1: Datos y recta ajustada
        ax1 = axes[0]
        ax1.scatter(X, y, alpha=0.6, s=60, edgecolors='k', color='lightblue', label='Datos')
        
        X_line = np.array([X.min(), X.max()])
        y_line = self.predict(X_line)
        ax1.plot(X_line, y_line, 'r-', linewidth=2.5, 
                label=f'Ajuste: y = {self.theta_0:.4f} + {self.theta_1:.4f}X')
        
        ax1.set_xlabel('X', fontsize=12, fontweight='bold')
        ax1.set_ylabel('y', fontsize=12, fontweight='bold')
        ax1.set_title('Regresión Lineal Ajustada' if title is None else title, 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Gráfica 2: Convergencia (coste vs iteraciones)
        ax2 = axes[1]
        if len(self.cost_history) > 0:
            ax2.plot(self.cost_history, linewidth=2, color='steelblue', label='Función de coste')
            ax2.scatter([len(self.cost_history)-1], [self.cost_history[-1]], 
                       color='red', s=100, zorder=5, label='Convergencia')
            ax2.set_xlabel('Iteraciones', fontsize=12, fontweight='bold')
            ax2.set_ylabel('J(θ)', fontsize=12, fontweight='bold')
            ax2.set_title('Convergencia del Gradiente Descendente', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def __repr__(self) -> str:
        """Representación en string del modelo."""
        return (f"LinearRegression(theta_0={self.theta_0:.6f}, "
                f"theta_1={self.theta_1:.6f})")


# Funciones de utilidad para uso directo (sin clase)

def linear_hypothesis(X: np.ndarray, theta_0: float, theta_1: float) -> np.ndarray:
    """
    Calcula la hipótesis lineal: h(X) = θ₀ + θ₁*X
    
    Esta es una función de utilidad que puede usarse directamente sin instanciar la clase.
    
    Args:
        X (np.ndarray): Variables independientes
        theta_0 (float): Parámetro de intercepto
        theta_1 (float): Parámetro de pendiente
    
    Returns:
        np.ndarray: Predicciones
    """
    return theta_0 + theta_1 * X


def compute_cost(X: np.ndarray, y: np.ndarray, theta_0: float, theta_1: float) -> float:
    """
    Calcula la función de coste: J(θ₀, θ₁) = (1/2m) * Σ(h(x) - y)²
    
    Esta es una función de utilidad que puede usarse directamente.
    
    Args:
        X (np.ndarray): Variables independientes
        y (np.ndarray): Variables dependientes
        theta_0 (float): Parámetro de intercepto
        theta_1 (float): Parámetro de pendiente
    
    Returns:
        float: Valor de la función de coste
    """
    m = len(X)
    h = linear_hypothesis(X, theta_0, theta_1)
    error = h - y
    J = (1 / (2 * m)) * np.sum(error ** 2)
    return J
