import numpy as np
from linreg_model import ajustar_modelo

if __name__ == "__main__":
    print("--- Ejemplo de Ejecución: Regresión Lineal ---")
    
    # Datos del laboratorio
    np.random.seed(42)
    X = np.linspace(0, 1, 100)
    y = 0.2 + 0.2*X + 0.02*np.random.random(100)

    # Ajustamos el modelo (esto generará las gráficas automáticamente)
    print("Entrenando el modelo...")
    resultados = ajustar_modelo(X, y, alpha=0.5, iteraciones=1000, graficar=True)

    print("\nResultados obtenidos:")
    print(f"Intercepto (theta_0): {resultados['theta_0']:.4f}")
    print(f"Pendiente (theta_1):  {resultados['theta_1']:.4f}")
    print(f"Coste Final (MSE):    {resultados['coste_final']:.6f}")
