
Actúa como un físico computacional experto en simulaciones de fluidos 2D.

Quiero un script COMPLETO en Python (un solo archivo) que simule una versión simplificada de la inestabilidad de Rayleigh–Taylor en 2D usando únicamente:

- numpy
- matplotlib
- matplotlib.animation

NO usar scipy ni otras dependencias.

Actúa como un físico computacional experto en dinámica de fluidos.

Quiero un script COMPLETO en Python (un solo archivo) que simule la inestabilidad de Rayleigh–Taylor 2D usando el modelo Boussinesq en formulación vorticidad–función de corriente.

Usar únicamente:
- numpy
- matplotlib
- matplotlib.animation
- El fondo de la animacion debe estar en negro


Necesito un script en Python que simule la inestabilidad de Rayleigh–Taylor en 2D usando el método de ‘stable fluids’ sobre una malla regular con celdas fantasma. El código debe: (1) resolver Navier–Stokes incompresibles bajo aproximación de Boussinesq con difusión, advección semilagrangiana, proyección y término de flotabilidad; (2) inicializar un perfil de densidad con fluido pesado arriba, ligero abajo y perturbación senoidal más ruido leve; (3) producir una animación con Matplotlib (FuncAnimation) del campo de densidad, mostrando el dominio vertical (Lx=0.25, Ly=1.0); Usar numpy y matplotlib, parámetros razonables (p.ej., NX=128, NY=256, DT=0.02, viscosidad y difusividad pequeñas).
