"""Herramientas para ajustar una regresion lineal simple por descenso por gradiente."""

from .model import (
    LinearRegressionResult,
    cost_function,
    fit_linear_regression,
    gradient_descent,
    linear_hypothesis,
)

__all__ = [
    "LinearRegressionResult",
    "cost_function",
    "fit_linear_regression",
    "gradient_descent",
    "linear_hypothesis",
]
