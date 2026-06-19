"""Example of installing and using the linear_reg_library package.

Installation:
    uv pip install .

Execution:
    uv run example_usage.py
"""

from __future__ import annotations

import numpy as np

from linear_reg_lib import fit_linear_regression


def main() -> None:
    """Run a simple example with the same ideal dataset pattern used in the lab."""
    x = np.arange(6, dtype=float)
    y = np.arange(6, dtype=float)

    model = fit_linear_regression(x, y, learning_rate=0.01, iterations=2000)

    print("Install with: uv pip install .")
    print("Run with: uv run example_usage.py")
    print(f"theta0: {model['theta0']:.6f}")
    print(f"theta1: {model['theta1']:.6f}")
    print(f"final cost: {model['cost']:.10f}")


if __name__ == "__main__":
    main()
