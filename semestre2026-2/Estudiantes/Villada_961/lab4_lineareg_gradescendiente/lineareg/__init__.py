"""Libreria de regresion lineal mediante coste y gradiente descendente."""

from .modelo import ajustar_modelo, coste, descenso_gradiente, hipotesis_lineal

__all__ = [
    "ajustar_modelo",
    "coste",
    "descenso_gradiente",
    "hipotesis_lineal",
]
