"""
Script de instalación para la librería linear_regression_gd

Uso:
    pip install .

Para instalación en modo desarrollo:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="linear_regression_gd",
    version="1.0.0",
    author="Estudiante de Física Computacional II",
    author_email="estudiante@udea.edu.co",
    description="Librería de Regresión Lineal con Gradiente Descendente",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/estudiante/linear_regression_gd",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
    ],
    keywords="machine-learning regression gradient-descent linear-regression",
    project_urls={
        "Bug Reports": "https://github.com/estudiante/linear_regression_gd/issues",
        "Source": "https://github.com/estudiante/linear_regression_gd",
        "Documentation": "https://github.com/estudiante/linear_regression_gd/blob/main/README.md",
    },
)
