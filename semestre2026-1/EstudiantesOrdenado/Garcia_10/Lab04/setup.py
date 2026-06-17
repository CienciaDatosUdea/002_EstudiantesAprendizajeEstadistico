"""
setup.py
--------
Archivo de configuración para instalar linreg_gd con pip.

Instalación
-----------
    Desde la carpeta Laboratorio_4/:

        pip install .

    O en modo editable (recomendado para desarrollo):

        pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name             = "linreg_gd",
    version          = "1.0.0",
    author           = "Garcia_10",
    description      = (
        "Librería para regresión lineal simple mediante "
        "función de coste y gradiente descendente."
    ),
    long_description = open("README.md", encoding="utf-8").read(),
    long_description_content_type = "text/markdown",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "numpy>=1.21",
    ],
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)