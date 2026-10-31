from setuptools import setup, find_packages

setup(
    name="linreg_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy"],
    author="Felipe Quitian",
    description="Librería básica para regresión lineal con gradiente descendente.",
    python_requires=">=3.6"
)
