"""Ejemplo de ajuste de regresion lineal con los datos del laboratorio."""

import numpy as np

from regresion_lineal_gd import fit_linear_regression


def main() -> None:
    """Ajusta una recta y muestra sus parametros."""
    generator = np.random.default_rng(42)
    x = np.linspace(0, 1, 100)
    y = 0.2 + 0.2 * x + 0.02 * generator.random(100)

    result = fit_linear_regression(x, y, learning_rate=0.5, tolerance=1e-10)
    print(f"Intercepto: {result.intercept:.6f}")
    print(f"Pendiente: {result.slope:.6f}")
    print(f"Coste J: {result.cost:.10f}")
    print(f"Iteraciones: {result.iterations}")


if __name__ == "__main__":
    main()
