"""Ejemplo de uso de la libreria sobre los datos del laboratorio."""

import numpy as np

from lab4_reglin import compute_cost, fit_linear_regression, linear_hypothesis


def main() -> None:
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = 0.2 + 0.2 * x + 0.02 * np.random.random(100)

    result = fit_linear_regression(x, y, alpha=0.5, n_iter=2000, tol=1e-10)

    print("Resultado del ajuste:")
    print(f"theta0: {result['theta0']:.6f}")
    print(f"theta1: {result['theta1']:.6f}")
    print(f"costo final: {result['cost']:.8f}")
    print(f"iteraciones: {result['iterations']}")

    y_hat = linear_hypothesis(x, result["theta0"], result["theta1"])
    check_cost = compute_cost(x, y, result["theta0"], result["theta1"])

    print(f"primeras 3 predicciones: {y_hat[:3]}")
    print(f"costo verificado: {check_cost:.8f}")


if __name__ == "__main__":
    main()
