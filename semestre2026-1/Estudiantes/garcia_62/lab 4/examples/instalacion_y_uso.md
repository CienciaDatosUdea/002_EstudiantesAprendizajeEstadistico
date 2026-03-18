# Ejemplo de instalacion y uso con pip

Ubicado en el directorio lab 4, ejecutar:

1. pip install -e .
2. python examples/ejemplo_laboratorio.py

Comandos completos en terminal:

cd "c:\Users\user\Desktop\Lab 1\002_EstudiantesAprendizajeEstadistico\semestre2026-1\Estudiantes\garcia_62\lab 4"
pip install -e .
python examples/ejemplo_laboratorio.py

Esto instala la libreria localmente y ejecuta un ajuste lineal usando:

X = np.linspace(0, 1, 100)
y = 0.2 + 0.2 * X + 0.02 * np.random.random(100)

El script imprime theta0, theta1 y el costo final.
