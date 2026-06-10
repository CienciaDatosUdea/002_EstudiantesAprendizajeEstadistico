Tarea de Prueba: Aproximación del valor de π usando C++ y funciones

1. Una forma clásica de aproximar el valor de π es mediante la **serie de Leibniz**, de convergencia lenta:

\begin{equation}
\pi = 4 \left( 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots \right)
\end{equation}

Otra forma de representar el mismo cálculo, separando explícitamente los términos positivos y negativos, es:

\begin{equation}
\pi = 4 \left( \sum_{k=0}^{n-1} \frac{(-1)^k}{2k+1} \right)
\end{equation}


Escribe un programa en C++ que aproxime el valor de π utilizando la **serie de Leibniz**, para ello:


- Crea una función `double calcularPiLeibniz(int n)`: que reciba el número de términos y devuelva el valor aproximado de π.


- Crea una función `double calcularPiSeparado(int n)`: que calcule π **sumando los términos con denominadores de la forma 4k+1 y restando los de 4k+3**, como se muestra en la segunda fórmula.


En la función main():

Solicitar al usuario el número de términos `n` a usar.
- Validar que `n` sea positivo.
- Calcular e imprimir ambas aproximaciones.
- Comparar con el valor real de π (`M_PI` en `<cmath>`).
- Mostrar el **error absoluto** en cada caso.

Requisitios:

- Código claro, modular y bien comentado.
- Validación básica de la entrada del usuario.
- Buenas prácticas: uso de funciones, separación lógica del código y claridad en las salidas.


Entrega:

La entrega debe realizarse a través del repositorio de GitHub asignado para el curso.

Los archivos requeridos son:

- `pi_leibniz.cpp`: archivo fuente con la implementación en C++.
- `pi_leibniz.out`: archivo ejecutable generado tras la compilación.
- `Guia_de_compilacion_y_ejecucion.md`: archivo en formato Markdown con una guía clara sobre cómo compilar y ejecutar el código en sistemas Linux.

Asegúrate de que el repositorio esté organizado y que los archivos se encuentren en el directorio principal o en una carpeta nombrada adecuadamente (`/ApellidoEstudiante/Tarea_00_pi/`, por ejemplo).