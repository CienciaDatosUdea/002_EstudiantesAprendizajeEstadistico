from setuptools import setup, find_packages

setup(
    name='linear_reg',
    version='0.1',
    description='Librería de Regresión Lineal por Gradiente Descendente',
    author='Jeronimo Lopez Gomez',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
)