# Guía de transición de Python a C++

## 1. Idea general

Python y C++ son lenguajes muy diferentes.

Python está diseñado para escribir código rápido, simple y legible. Muchas cosas se manejan automáticamente: tipos de datos, memoria, tamaños de variables, estructuras dinámicas, etc.

C++ exige ser más explícito. El programador debe declarar tipos de variables, compilar el programa antes de ejecutarlo y tener más cuidado con memoria, referencias, punteros y eficiencia.

En resumen:

| Python | C++ |
|---|---|
| Interpretado | Compilado |
| Tipado dinámico | Tipado estático |
| Sintaxis simple | Sintaxis más estricta |
| Manejo automático de memoria | Más control sobre memoria |
| Muy usado en ciencia de datos, scripts e IA | Muy usado en simulación, sistemas, videojuegos, física computacional y alto rendimiento |

---

## 2. Primer programa

### Python

```python
print("Hola mundo")
```

### C++

```cpp
#include <iostream>

int main() {
    std::cout << "Hola mundo" << std::endl;
    return 0;
}
```

En C++ todo programa ejecutable necesita una función principal:

```cpp
int main() {
    // código
    return 0;
}
```

---

## 3. Compilar y ejecutar

En Python normalmente ejecutamos:

```bash
python programa.py
```

En C++ primero compilamos:

```bash
g++ programa.cpp -o programa
```

Luego ejecutamos:

```bash
./programa
```

Ejemplo completo:

```bash
g++ main.cpp -o main
./main
```

---

## 4. Variables y tipos de datos

En Python no se declara el tipo:

```python
x = 10
y = 3.14
nombre = "Ana"
```

En C++ sí se debe declarar:

```cpp
int x = 10;
double y = 3.14;
std::string nombre = "Ana";
```

Tipos básicos en C++:

```cpp
int edad = 20;              // entero
float altura = 1.75;        // decimal simple
double pi = 3.141592;       // decimal más preciso
char letra = 'A';           // carácter
bool activo = true;         // booleano
std::string nombre = "Ana"; // texto
```

Para usar `std::string` se incluye:

```cpp
#include <string>
```

---

## 5. Salida por pantalla

### Python

```python
print("Hola", nombre)
```

### C++

```cpp
std::cout << "Hola " << nombre << std::endl;
```

Ejemplo:

```cpp
#include <iostream>
#include <string>

int main() {
    std::string nombre = "Ana";
    std::cout << "Hola " << nombre << std::endl;
    return 0;
}
```

---

## 6. Entrada por teclado

### Python

```python
nombre = input("Ingrese su nombre: ")
```

### C++

```cpp
std::string nombre;
std::cout << "Ingrese su nombre: ";
std::cin >> nombre;
```

Ejemplo completo:

```cpp
#include <iostream>
#include <string>

int main() {
    std::string nombre;

    std::cout << "Ingrese su nombre: ";
    std::cin >> nombre;

    std::cout << "Hola " << nombre << std::endl;

    return 0;
}
```

Para leer frases con espacios:

```cpp
std::getline(std::cin, nombre);
```

---

## 7. Condicionales

### Python

```python
x = 10

if x > 0:
    print("positivo")
elif x < 0:
    print("negativo")
else:
    print("cero")
```

### C++

```cpp
int x = 10;

if (x > 0) {
    std::cout << "positivo" << std::endl;
} else if (x < 0) {
    std::cout << "negativo" << std::endl;
} else {
    std::cout << "cero" << std::endl;
}
```

Diferencias importantes:

- En Python se usan dos puntos `:`.
- En C++ se usan llaves `{}`.
- En C++ las condiciones van entre paréntesis.
- En C++ cada instrucción termina en punto y coma `;`.

---

## 8. Ciclo `for`

### Python

```python
for i in range(5):
    print(i)
```

Salida:

```text
0
1
2
3
4
```

### C++

```cpp
for (int i = 0; i < 5; i++) {
    std::cout << i << std::endl;
}
```

Estructura general:

```cpp
for (inicialización; condición; actualización) {
    // código
}
```

Ejemplo:

```cpp
for (int i = 1; i <= 10; i++) {
    std::cout << i << std::endl;
}
```

---

## 9. Ciclo `while`

### Python

```python
i = 0

while i < 5:
    print(i)
    i += 1
```

### C++

```cpp
int i = 0;

while (i < 5) {
    std::cout << i << std::endl;
    i++;
}
```

---

## 10. Listas y arreglos

En Python usamos listas:

```python
numeros = [1, 2, 3, 4]
print(numeros[0])
```

En C++ hay varias opciones. La más recomendable para empezar es `std::vector`.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> numeros = {1, 2, 3, 4};

    std::cout << numeros[0] << std::endl;

    return 0;
}
```

Agregar elementos:

### Python

```python
numeros.append(5)
```

### C++

```cpp
numeros.push_back(5);
```

Recorrer un vector:

```cpp
for (int i = 0; i < numeros.size(); i++) {
    std::cout << numeros[i] << std::endl;
}
```

También se puede hacer:

```cpp
for (int n : numeros) {
    std::cout << n << std::endl;
}
```

---

## 11. Funciones

### Python

```python
def suma(a, b):
    return a + b

resultado = suma(3, 4)
print(resultado)
```

### C++

```cpp
#include <iostream>

int suma(int a, int b) {
    return a + b;
}

int main() {
    int resultado = suma(3, 4);
    std::cout << resultado << std::endl;

    return 0;
}
```

En C++ se debe indicar:

- El tipo que retorna la función.
- El tipo de cada parámetro.

Ejemplo:

```cpp
double area_circulo(double r) {
    return 3.1416 * r * r;
}
```

---

## 12. Comentarios

### Python

```python
# Esto es un comentario
```

### C++

```cpp
// Esto es un comentario de una línea
```

Comentario de varias líneas:

```cpp
/*
Esto es un comentario
de varias líneas
*/
```

---

## 13. Operadores

Son muy similares:

| Operación | Python | C++ |
|---|---|---|
| Suma | `a + b` | `a + b` |
| Resta | `a - b` | `a - b` |
| Multiplicación | `a * b` | `a * b` |
| División | `a / b` | `a / b` |
| Módulo | `a % b` | `a % b` |
| Potencia | `a ** b` | `pow(a, b)` |

En C++ para usar `pow`:

```cpp
#include <cmath>
```

Ejemplo:

```cpp
#include <iostream>
#include <cmath>

int main() {
    double x = pow(2, 3);
    std::cout << x << std::endl;

    return 0;
}
```

---

## 14. Cuidado con la división entera

En Python:

```python
print(5 / 2)
```

Resultado:

```text
2.5
```

En C++:

```cpp
std::cout << 5 / 2 << std::endl;
```

Resultado:

```text
2
```

Porque ambos números son enteros.

Para obtener decimal:

```cpp
std::cout << 5.0 / 2.0 << std::endl;
```

O usando variables `double`:

```cpp
double a = 5;
double b = 2;

std::cout << a / b << std::endl;
```

---

## 15. Módulos en Python vs librerías en C++

En Python importamos:

```python
import math
```

En C++ incluimos cabeceras:

```cpp
#include <cmath>
```

Ejemplos comunes:

```cpp
#include <iostream>  // entrada y salida
#include <string>    // textos
#include <vector>    // vectores
#include <cmath>     // funciones matemáticas
#include <fstream>   // archivos
```

---

## 16. Archivos

### Leer archivo en Python

```python
with open("datos.txt", "r") as archivo:
    contenido = archivo.read()
    print(contenido)
```

### Leer archivo en C++

```cpp
#include <iostream>
#include <fstream>
#include <string>

int main() {
    std::ifstream archivo("datos.txt");
    std::string linea;

    while (std::getline(archivo, linea)) {
        std::cout << linea << std::endl;
    }

    archivo.close();

    return 0;
}
```

### Escribir archivo en C++

```cpp
#include <iostream>
#include <fstream>

int main() {
    std::ofstream archivo("salida.txt");

    archivo << "Hola desde C++" << std::endl;

    archivo.close();

    return 0;
}
```

---

## 17. Clases

### Python

```python
class Persona:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad

    def saludar(self):
        print("Hola, soy", self.nombre)
```

### C++

```cpp
#include <iostream>
#include <string>

class Persona {
private:
    std::string nombre;
    int edad;

public:
    Persona(std::string n, int e) {
        nombre = n;
        edad = e;
    }

    void saludar() {
        std::cout << "Hola, soy " << nombre << std::endl;
    }
};

int main() {
    Persona p("Ana", 20);
    p.saludar();

    return 0;
}
```

En C++ aparecen tres ideas importantes:

```cpp
private:
```

Elementos que solo se usan dentro de la clase.

```cpp
public:
```

Elementos accesibles desde fuera de la clase.

```cpp
Persona(std::string n, int e)
```

Constructor de la clase.

---

## 18. Punteros

En Python casi nunca pensamos explícitamente en direcciones de memoria.

En C++ sí podemos trabajar con direcciones.

```cpp
int x = 10;
int* p = &x;

std::cout << x << std::endl;   // valor de x
std::cout << &x << std::endl;  // dirección de x
std::cout << p << std::endl;   // dirección guardada en p
std::cout << *p << std::endl;  // valor apuntado por p
```

Significado:

| Símbolo | Significado |
|---|---|
| `&x` | Dirección de memoria de `x` |
| `int* p` | Puntero a entero |
| `*p` | Valor almacenado en la dirección apuntada |

Este tema es fundamental en C++, pero puede estudiarse después de dominar variables, funciones, arreglos y clases.

---

## 19. Referencias

Una referencia es como otro nombre para una variable.

```cpp
int x = 10;
int& ref = x;

ref = 20;

std::cout << x << std::endl;
```

Salida:

```text
20
```

Las referencias son muy usadas para evitar copias innecesarias.

Ejemplo:

```cpp
void duplicar(int& x) {
    x = 2 * x;
}
```

---

## 20. Comparación rápida de sintaxis

| Concepto | Python | C++ |
|---|---|---|
| Imprimir | `print(x)` | `std::cout << x << std::endl;` |
| Leer | `input()` | `std::cin >> x;` |
| Entero | `x = 5` | `int x = 5;` |
| Decimal | `x = 3.14` | `double x = 3.14;` |
| Texto | `s = "hola"` | `std::string s = "hola";` |
| Lista | `[1,2,3]` | `std::vector<int> v = {1,2,3};` |
| Función | `def f(x):` | `int f(int x) {}` |
| Clase | `class A:` | `class A {};` |
| Comentario | `# comentario` | `// comentario` |

---

## 21. Ejemplo completo: promedio de notas

### Python

```python
notas = [4.0, 3.5, 5.0, 2.8]

suma = 0

for nota in notas:
    suma += nota

promedio = suma / len(notas)

print("Promedio:", promedio)
```

### C++

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<double> notas = {4.0, 3.5, 5.0, 2.8};

    double suma = 0.0;

    for (double nota : notas) {
        suma += nota;
    }

    double promedio = suma / notas.size();

    std::cout << "Promedio: " << promedio << std::endl;

    return 0;
}
```

---

## 22. Ejemplo completo: movimiento con Euler

Este ejemplo es útil para física computacional.

Queremos resolver:

$$
\frac{dx}{dt} = v
$$

con velocidad constante.

### Python

```python
x = 0.0
v = 2.0
dt = 0.1
N = 10

for i in range(N):
    x = x + v * dt
    print(i, x)
```

### C++

```cpp
#include <iostream>

int main() {
    double x = 0.0;
    double v = 2.0;
    double dt = 0.1;
    int N = 10;

    for (int i = 0; i < N; i++) {
        x = x + v * dt;
        std::cout << i << " " << x << std::endl;
    }

    return 0;
}
```

---

## 23. Errores comunes al pasar de Python a C++

### Olvidar el punto y coma

Incorrecto:

```cpp
int x = 10
```

Correcto:

```cpp
int x = 10;
```

### Olvidar incluir librerías

Incorrecto:

```cpp
std::cout << "Hola";
```

si no se ha incluido:

```cpp
#include <iostream>
```

### Usar variables sin declararlas

Incorrecto:

```cpp
x = 10;
```

Correcto:

```cpp
int x = 10;
```

### Confundir `=` con `==`

Asignación:

```cpp
x = 5;
```

Comparación:

```cpp
if (x == 5) {
    std::cout << "x vale 5" << std::endl;
}
```

### División entera accidental

```cpp
int a = 5;
int b = 2;

std::cout << a / b << std::endl; // da 2
```

Para obtener `2.5`:

```cpp
double a = 5;
double b = 2;

std::cout << a / b << std::endl;
```


## 25. Ejercicios propuestos

### Ejercicio 1

Escribir un programa en C++ que imprima:

```text
Hola, estoy aprendiendo C++
```

### Ejercicio 2

Pedir al usuario dos números y mostrar la suma.

### Ejercicio 3

Pedir un número entero y decir si es par o impar.

### Ejercicio 4

Calcular el promedio de cinco notas almacenadas en un vector.

### Ejercicio 5

Crear una función que reciba un número y retorne su cuadrado.

### Ejercicio 6

Simular el movimiento de una partícula con velocidad constante usando el método de Euler.

### Ejercicio 7

Leer una lista de números desde un archivo y calcular su promedio.

### Ejercicio 8

Crear una clase `Particula` con posición, velocidad y un método para actualizar su posición.

