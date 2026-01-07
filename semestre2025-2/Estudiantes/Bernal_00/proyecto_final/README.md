# Aprendizaje Automático en el Modelo de Ising 2D

Este proyecto implementa métodos de aprendizaje automático para el análisis y clasificación de fases en el modelo de Ising bidimensional. El trabajo combina simulaciones Monte Carlo con redes neuronales profundas para identificar transiciones de fase y reconstruir configuraciones de espines.

## Descripción del Proyecto

El modelo de Ising es un modelo fundamental en física estadística que describe sistemas magnéticos mediante una red de espines que pueden tomar valores +Reconstrucción de configuraciones (inpainting)

Se propone reformular el problema desde una tarea de clasificación de fases, en la que se asigna una etiqueta global a cada configuración del sistema, hacia una tarea de reconstrucción parcial (inpainting o imputación). En este enfoque, se oculta de manera controlada un pequeño porcentaje de los espines de cada configuración (por ejemplo, un 5%) y se entrena un modelo para predecir los valores de los espines ocultos a partir del resto de la configuración observada.

Este procedimiento permite evaluar en qué medida el modelo ha capturado las correlaciones espaciales intrínsecas del sistema de Ising, ya que la reconstrucción correcta de los espines faltantes depende directamente del conocimiento implícito de dichas correlaciones. En particular, la dificultad de la tarea de reconstrucción, cuantificada mediante métricas de error adecuadas, puede utilizarse como un indicador sensible de las propiedades físicas del sistema, incluyendo cambios estructurales asociados a transiciones de fase.1 o -1. En dos dimensiones, el modelo presenta una transición de fase de segundo orden a una temperatura crítica T_c ≈ 2.269, separando una fase ordenada (T < T_c) de una fase desordenada (T > T_c).

Este proyecto aborda tres tareas principales:

1. **Generación de datos**: Implementación del algoritmo de Metropolis-Hastings para generar configuraciones de equilibrio del modelo de Ising a diferentes temperaturas.
2. **Clasificación de fases**: Entrenamiento de redes neuronales para clasificar automáticamente las configuraciones en fase ordenada o desordenada.
3. **Reconstrucción de espines**: Implementación de una arquitectura U-Net para reconstruir configuraciones parcialmente ocultas (inpainting), permitiendo medir correlaciones aprendidas por el modelo.

## Estructura del Proyecto

```
proyecto_final/
├── data/
├── models/
│   ├── CNN_classification.ipynb
│   ├── CNN_spin_reconstruction.ipynb
│   ├── NN_classification.ipynb
│   └── experiments/
├── Ising_generator.ipynb
├── Distribuciones.ipynb
├── utils.py
└── README.md
```

## Generación de Datos (Ising_generator.ipynb)

### Algoritmo de Metropolis-Hastings

El generador implementa el algoritmo de Metropolis-Hastings para muestrear configuraciones del ensamble canónico:

1. **Inicialización**: Red cuadrada L×L con espines aleatorios (±1)
2. **Simulated Annealing**: Barrido de temperatura desde T=5 hasta la temperatura objetivo para evitar mínimos locales
3. **Burn-in (Termalización)**: Ejecución de pasos Monte Carlo sin guardar datos para alcanzar equilibrio térmico
4. **Muestreo**: Guardado de configuraciones espaciadas en el tiempo para reducir correlaciones

### Formato de Almacenamiento: HDF5 (H5)

En el proyecto se optó por almacenar los datos en disco usando el formato HDF5 (h5) y cargarlos de manera incremental por batches durante el entrenamiento, en lugar de mantener todo el conjunto de datos en memoria RAM. Esta decisión se fundamenta en varias razones técnicas y prácticas:

#### 1. Tamaño del Conjunto de Datos

Las configuraciones del modelo de Ising consisten en arreglos bidimensionales de espines, y el número total de muestras crece rápidamente al considerar múltiples temperaturas, diferentes realizaciones independientes (Monte Carlo), y posibles promedios o réplicas. Cargar el dataset completo en memoria no escala bien y puede exceder fácilmente la capacidad de RAM disponible, especialmente durante el entrenamiento de redes neuronales. El uso de HDF5 permite manejar datasets grandes sin requerir que estén completamente en memoria.

#### 2. Eficiencia de I/O y Acceso Parcial

El formato HDF5 está diseñado para acceso eficiente a subconjuntos de datos, lo cual es ideal para lectura por índices, slicing por rangos, y acceso directo a batches consecutivos o aleatorios. Esto hace que la carga de datos por batches sea rápida y estable, minimizando el cuello de botella de entrada/salida (I/O) durante el entrenamiento.

#### 3. Compatibilidad Natural con el Entrenamiento por Mini-batches

El entrenamiento de redes neuronales se realiza típicamente mediante mini-batch gradient descent, por lo que no es necesario tener todos los datos simultáneamente en memoria; basta con disponer de un batch a la vez. Cargar los datos en batches desde disco reduce el uso de memoria, mantiene constante el consumo de recursos, y permite entrenar modelos más grandes o usar datasets más extensos.

#### 4. Separación entre Generación de Datos y Entrenamiento

En el modelo de Ising, la generación de configuraciones (simulación Monte Carlo) es computacionalmente costosa y conceptualmente distinta del entrenamiento del modelo de ML. Al guardar las configuraciones en archivos HDF5, la simulación se realiza una sola vez, los datos pueden reutilizarse múltiples veces, y se facilita la reproducibilidad del experimento. Esto permite iterar sobre la arquitectura o los hiperparámetros del modelo sin volver a simular el sistema físico.

#### 5. Escalabilidad y Reproducibilidad

Este enfoque permite escalar el número de muestras sin cambiar la lógica del entrenamiento, entrenar en máquinas con distinta capacidad de memoria, y mantener un formato de datos estructurado y documentado (grupos, metadatos, etiquetas). Además, HDF5 facilita almacenar temperaturas, magnetización, energía, y etiquetas de fase junto con las configuraciones de espines.

#### 6. Práctica Estándar en Proyectos Científicos y de ML

El uso de almacenamiento en disco con carga por batches es una práctica estándar en simulaciones numéricas, física computacional, y aprendizaje automático con datasets grandes. Por tanto, esta elección no solo es pragmática, sino también alineada con buenas prácticas profesionales.

### Estructura de Archivos Generados

Cada archivo de datos contiene:

- **Header (primera línea)**: Metadatos en formato JSON con parámetros de simulación:
  ```json
  {"L": 10, "T": 2.3, "N": 5000, "class": 1, "burn_in": 2000, "interval": 100, "seed": 42, "H": 0, "J": 1}
  ```
- **Configuraciones**: Cada línea subsiguiente contiene un microestado plano (valores separados por comas)

El nombre del archivo sigue el patrón: `ising_L{L}_T{T:.3f}_{label}.txt`, donde `label` es 0 para fase ordenada y 1 para fase desordenada.

## Modelos Implementados

### 1. Red Neuronal Fully Connected (NN_classification.ipynb)

Arquitectura de red neuronal densa para clasificación binaria de fases:

- **Arquitectura**: Secuencia de capas fully connected con activaciones ReLU
- **Topología**: Input → 100 → 100 → 200 → 100 → 50 → 1
- **Función de pérdida**: Binary Cross-Entropy with Logits Loss
- **Optimizador**: Adam con learning rate 1e-3
- **Aplicación**: Clasificación de configuraciones en fase ordenada (0) o desordenada (1)

### 2. Red Neuronal Convolucional (CNN_classification.ipynb)

Arquitectura CNN basada en AlexNet adaptada para configuraciones de Ising:

- **Arquitectura**: 
  - Capas convolucionales: Conv2d(1→32→64→128) con MaxPooling
  - Capas fully connected: 512 → 256 → 128 → 1
  - Dropout (0.5) para regularización
- **Función de pérdida**: Binary Cross-Entropy with Logits Loss
- **Optimizador**: Adam con learning rate 1e-3
- **Ventaja**: Captura patrones espaciales locales y correlaciones entre espines vecinos

### 3. U-Net para Reconstrucción de Espines (CNN_spin_reconstruction.ipynb)

Arquitectura U-Net para tarea de inpainting/reconstrucción:

- **Arquitectura**: 
  - Encoder: Bloques DoubleConv con MaxPooling (64→128→256→512 canales)
  - Decoder: Bloques Up con concatenación skip connections (512→256→128→64)
  - Entrada: 2 canales (estado enmascarado + máscara)
  - Salida: 1 canal (probabilidad de spin up)
- **Función de pérdida**: Binary Cross-Entropy with Logits Loss (aplicada solo sobre píxeles enmascarados)
- **Optimizador**: Adam con learning rate 1e-3
- **Aplicación**: Reconstrucción de 5% de espines ocultos aleatoriamente
- **Métrica de interés**: Masked Negative Log-Likelihood (NLL) como función de la temperatura, mostrando pico cerca de T_c

## Utilidades (utils.py)

El módulo `utils.py` proporciona las siguientes clases y funcionalidades:

- **`read_ising_file`**: Clase para leer y procesar archivos de datos del modelo Ising, incluyendo métodos para visualización, cálculo de energía y magnetización.
- **`ising_data_builder`**: Construye datasets en formato HDF5 a partir de archivos de texto, soportando dos modos:
  - `kind='classification'`: Genera dataset con etiquetas de fase
  - `kind='inpainting'`: Genera dataset para tarea de reconstrucción
- **`H5IsingDataset`**: Dataset de PyTorch que permite cargar datos desde archivos HDF5 de manera eficiente, compatible con DataLoader para entrenamiento por batches.

## Análisis de Distribuciones (Distribuciones.ipynb)

Este notebook analiza las propiedades estadísticas del modelo de Ising:

- **Distribuciones de energía**: Histogramas de energía para diferentes temperaturas
- **Distribuciones de magnetización**: Histogramas de magnetización para diferentes temperaturas
- **Magnetización promedio vs temperatura**: Curva que muestra la transición de fase

## Requisitos

Las principales dependencias del proyecto incluyen:

- Python 3.x
- NumPy
- PyTorch
- Matplotlib
- Seaborn
- scikit-learn
- h5py
- tqdm
- Numba (para optimización del generador)

## Uso

### Generación de Datos

```python
from Ising_generator import Ising_2D
import numpy as np

ising = Ising_2D()
T = np.arange(0.1, 4.1, 0.1)

for t in T:
    ising.generate_samples(
        L=10, 
        T=t, 
        samples=2000, 
        burn_in=5000, 
        interval=500, 
        seed=73, 
        H=0, 
        J=1, 
        folder='./data/data_10/'
    )
```

### Construcción del Dataset HDF5

```python
from utils import ising_data_builder, H5IsingDataset

# Para clasificación
data = ising_data_builder('./data/data_10/', kind='classification').h5_path
dataset = H5IsingDataset(data)

# Para inpainting
data = ising_data_builder('./data/data_30/', kind='inpainting').h5_path
dataset = H5IsingDataset(data)
```

### Entrenamiento de Modelos

Los notebooks en la carpeta `models/` contienen el código completo para entrenar cada modelo. Cada notebook incluye:

- Construcción y división del dataset (train/test)
- Definición de la arquitectura del modelo
- Loop de entrenamiento con métricas
- Evaluación y visualización de resultados

## Resultados

Los modelos implementados permiten:

1. **Clasificación precisa de fases**: Las redes neuronales logran alta precisión en la clasificación de configuraciones, con accuracy que varía según la temperatura, mostrando mayor dificultad cerca de la temperatura crítica T_c.

2. **Detección de transición de fase mediante inpainting**: El modelo U-Net muestra un pico en la Masked NLL cerca de T_c, indicando que la dificultad de reconstrucción aumenta en la región crítica, donde las correlaciones son más complejas.

3. **Análisis de propiedades físicas**: Las distribuciones de energía y magnetización revelan el comportamiento característico del modelo de Ising en diferentes regímenes de temperatura.
