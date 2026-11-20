import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar las funciones desde la librería creada
from regresion_lineal import fit_linear_regression, predict, cost_function

def main():
    print("--- Demostración de la librería de Regresión Lineal ---")
    
    # 1. Crear datos sintéticos simples (y = 2x + 1 con pequeño ruido)
    np.random.seed(42)
    x_data = np.linspace(0, 10, 30)
    y_data = 2.0 * x_data + 1.0 + np.random.normal(0, 0.3, size=len(x_data))
    
    df = pd.DataFrame({'X': x_data, 'y': y_data})
    
    # 2. Ajustar el modelo usando la función principal
    print("\nAjustando modelo con fit_linear_regression...")
    theta0, theta1, history = fit_linear_regression(
        df['X'], df['y'], alpha=0.01, epsilon=1e-14, max_iter=100000
    )
    
    print(f"\nResultados del modelo ajustado:")
    print(f"  - theta0 (Intercepto): {theta0:.6f}")
    print(f"  - theta1 (Pendiente):  {theta1:.6f}")
    print(f"  - Coste final:        {history[-1]:.8f}")
    print(f"  - Iteraciones:        {len(history)}")
    
    # 3. Predicción
    y_pred = predict(df['X'], theta0, theta1)
    
    # 4. Visualización
    plt.figure(figsize=(8, 5))
    plt.scatter(df['X'], df['y'], color='blue', label='Datos reales')
    plt.plot(df['X'], y_pred, color='red', linewidth=2, label=f'Recta: y = {theta0:.2f} + {theta1:.2f}x')
    plt.title('Ajuste de Regresión Lineal con la Librería')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('resultado_libreria.png')
    plt.show()

if __name__ == "__main__":
    main()