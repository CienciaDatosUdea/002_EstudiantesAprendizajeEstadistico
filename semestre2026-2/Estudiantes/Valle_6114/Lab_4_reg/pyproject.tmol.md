[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "regresion_lineal_gd"
version = "0.1.0"
authors = [
    { name="Estudiante Lab", email="estudiante@ejemplo.com" }
]
description = "Librería simple para regresión lineal por Descenso de Gradiente"
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0"
]

[project.urls]
"Homepage" = "https://github.com/ejemplo/regresion_lineal_gd"