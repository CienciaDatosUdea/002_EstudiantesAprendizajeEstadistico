import numpy as np
import matplotlib.pylab as plots

def regresion_lineal(r):
    """Genera un gráfico de regresión lineal en unidades estándar para un valor dado de r."""
    
    # Generar datos aleatorios
    np.random.seed(42)
    n_samples = 1000
    x = np.random.normal(size=n_samples)
    y = r * x + np.random.normal(scale=np.sqrt(1 - r**2), size=n_samples)  # Asegura que varianza total sea 1

    # Estandarizar los datos manualmente
    x_standardized = (x - np.mean(x)) / np.std(x)
    y_standardized = (y - np.mean(y)) / np.std(y)

    # Crear el gráfico
    plots.figure(figsize=(6,6))
    plots.scatter(x_standardized, y_standardized, color='blue', alpha=0.6, s=10, label="Datos")
    
    # Dibujar la recta de regresión y = r * x
    x_line = np.linspace(-4, 4, 100)
    plots.plot(x_line, r * x_line, color='red', linewidth=2, label=f'Regresión (pendiente = {r})')

    # Dibujar la línea de identidad y = x
    plots.plot(x_line, x_line, color='green', linestyle='dashed', linewidth=2, label="Línea identidad (y = x)")

    # Etiquetas y título
    plots.xlabel("x en unidades estándar")
    plots.ylabel("y en unidades estándar")
    plots.title(f"Regresión Lineal con r = {r}, r² = {r**2:.2f}")
    plots.legend()
    plots.grid(True)
    
    # Mostrar el gráfico
    plots.show()

