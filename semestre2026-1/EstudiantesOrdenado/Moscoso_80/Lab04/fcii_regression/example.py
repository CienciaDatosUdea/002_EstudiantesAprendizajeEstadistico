"""
Ejemplo de uso de la librería fcii_regression.

Este script demuestra cómo usar la librería para:
1. Generar datos sintéticos con relación lineal
2. Configurar el optimizador de Gradiente Descendente
3. Entrenar un modelo de Regresión Lineal
4. Visualizar resultados y diagnósticos

Ejecutar:
    python example.py

Nota: Antes de ejecutar, instalar la librería con:
    pip install -e .
"""

import numpy as np
import matplotlib.pyplot as plt

# Importar la librería
from fcii_regression import GradientDescent, LinearRegression


def main():
    """Función principal del ejemplo."""

    print("=" * 60)
    print("EJEMPLO: Regresión Lineal con Gradiente Descendente")
    print("Librería: fcii_regression")
    print("=" * 60)

    # =========================================================================
    # 1. GENERAR DATOS SINTÉTICOS
    # =========================================================================
    print("\n[1] Generando datos sintéticos...")

    np.random.seed(42)  # Reproducibilidad

    # Parámetros verdaderos de la relación lineal
    theta_0_real = 1.5   # Intercepto
    theta_1_real = 2.3   # Pendiente
    noise_std = 2.0      # Desviación del ruido

    # Generar datos
    n_samples = 50
    X = np.linspace(0, 10, n_samples)
    y = theta_0_real + theta_1_real * X + np.random.randn(n_samples) * noise_std

    print(f"   - Ecuación real: y = {theta_0_real} + {theta_1_real} * x + ruido")
    print(f"   - Número de muestras: {n_samples}")
    print(f"   - Rango de X: [{X.min():.1f}, {X.max():.1f}]")

    # =========================================================================
    # 2. CONFIGURAR EL OPTIMIZADOR
    # =========================================================================
    print("\n[2] Configurando Gradiente Descendente...")

    gd = GradientDescent(
        n_iterations=2000,    # Número de iteraciones
        learning_rate=0.01,   # Tasa de aprendizaje
        h=0.001,              # Paso para diferencias finitas
        verbose=True          # Imprimir progreso
    )

    print(f"   - Iteraciones: {gd.n_iterations}")
    print(f"   - Learning rate: {gd.learning_rate}")
    print(f"   - Paso h: {gd.h}")

    # =========================================================================
    # 3. ENTRENAR EL MODELO
    # =========================================================================
    print("\n[3] Entrenando modelo de Regresión Lineal...")
    print("-" * 60)

    model = LinearRegression(optimizer=gd)
    model.fit(X, y, initial_theta=[0.0, 0.0])  # Iniciar en origen

    print("-" * 60)

    # =========================================================================
    # 4. MOSTRAR RESULTADOS
    # =========================================================================
    print("\n[4] Resultados del entrenamiento:")
    print(f"   - Ecuación encontrada: {model.get_equation()}")
    print(f"   - Parámetros: θ₀ = {model.theta_[0]:.4f}, θ₁ = {model.theta_[1]:.4f}")
    print(f"   - Coeficiente R²: {model.score(X, y):.4f}")
    print(f"   - MSE final: {model.cost_history_[-1]:.6f}")

    # Comparación con valores reales
    print("\n   Comparación con valores reales:")
    print(f"   - θ₀ real: {theta_0_real:.4f}, θ₀ estimado: {model.theta_[0]:.4f}, "
          f"error: {abs(theta_0_real - model.theta_[0]):.4f}")
    print(f"   - θ₁ real: {theta_1_real:.4f}, θ₁ estimado: {model.theta_[1]:.4f}, "
          f"error: {abs(theta_1_real - model.theta_[1]):.4f}")

    # =========================================================================
    # 5. VISUALIZACIONES
    # =========================================================================
    print("\n[5] Generando visualizaciones...")

    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 10))

    # 5.1 Ajuste del modelo
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(X, y, color='blue', alpha=0.6, label='Datos', edgecolors='black')
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)
    ax1.plot(X_line, y_line, color='red', linewidth=2,
             label=f'Ajuste: {model.get_equation()}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Regresión Lineal - Ajuste del Modelo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 5.2 Convergencia del coste
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(model.cost_history_, color='blue', linewidth=1)
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('J(θ) - MSE')
    ax2.set_title('Convergencia del Gradiente Descendente')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=model.cost_history_[-1], color='red', linestyle='--',
                label=f'Coste final = {model.cost_history_[-1]:.4f}')
    ax2.legend()

    # 5.3 Residuos
    ax3 = fig.add_subplot(2, 2, 3)
    y_pred = model.predict(X)
    residuals = y - y_pred
    ax3.scatter(y_pred, residuals, color='green', alpha=0.6, edgecolors='black')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Valores predichos (ŷ)')
    ax3.set_ylabel('Residuos (y - ŷ)')
    ax3.set_title('Análisis de Residuos')
    ax3.grid(True, alpha=0.3)

    # 5.4 Histograma de residuos
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(residuals, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='red', linestyle='--', label='Residuo = 0')
    ax4.set_xlabel('Residuos')
    ax4.set_ylabel('Frecuencia')
    ax4.set_title('Distribución de Residuos')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('example_results.png', dpi=150)
    print("   - Gráfica guardada: example_results.png")
    plt.show()

    # 5.5 Superficie 3D del coste
    print("\n   Generando superficie 3D de la función de coste...")
    fig2, ax_3d = model.plot_cost_surface(
        theta0_range=(-2, 5),
        theta1_range=(0, 5),
        resolution=50,
        title="Superficie de la Función de Coste J(θ₀, θ₁)"
    )
    plt.tight_layout()
    plt.savefig('cost_surface_3d.png', dpi=150)
    print("   - Gráfica guardada: cost_surface_3d.png")
    plt.show()

    # 5.6 Mapa de contorno
    print("\n   Generando mapa de contorno...")
    fig3, ax_contour = model.plot_cost_contour(
        theta0_range=(-2, 5),
        theta1_range=(0, 5),
        resolution=100,
        title="Mapa de Contorno de J(θ₀, θ₁)"
    )
    plt.tight_layout()
    plt.savefig('cost_contour.png', dpi=150)
    print("   - Gráfica guardada: cost_contour.png")
    plt.show()

    # =========================================================================
    # 6. PREDICCIONES
    # =========================================================================
    print("\n[6] Ejemplo de predicciones:")
    X_new = np.array([2.5, 5.0, 7.5, 10.0])
    y_new = model.predict(X_new)

    print("   X      | ŷ (predicción)")
    print("   " + "-" * 25)
    for x_val, y_val in zip(X_new, y_new):
        print(f"   {x_val:.1f}    | {y_val:.4f}")

    print("\n" + "=" * 60)
    print("Ejemplo completado exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
