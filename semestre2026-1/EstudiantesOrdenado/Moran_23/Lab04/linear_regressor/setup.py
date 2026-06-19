# Contenido de setup.py
from setuptools import setup, find_packages
 
setup(
    name='linreg',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy'],
    description='Regresion lineal con gradiente descendente',
    author='Tu Nombre',
)
