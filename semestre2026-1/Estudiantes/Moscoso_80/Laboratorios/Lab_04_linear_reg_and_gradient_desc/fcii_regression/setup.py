"""
Setup script para instalar fcii_regression con pip.

Uso:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fcii_regression",
    version="1.0.0",
    author="Estudiante FCII - UdeA",
    author_email="estudiante@udea.edu.co",
    description="Librería de Regresión Lineal con Gradiente Descendente para FCII",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/udea/fcii-regression",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "jupyter>=1.0",
        ],
    },
)
