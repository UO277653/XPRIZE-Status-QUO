Historial de cambios pre creación repositorio
============================================
> v1.0 Versión inicial: ejemplo de la página de Pennylane
>
Tareas siguientes <br>
Intento aplicar VQLS al problema pero es necesario obtener una descomposición

> v1.1 VPFS y traducción de Qiskit a Pennylane

Tareas siguientes <br>
~~Poder intercambiar ansatz más fácilmente~~ (quedaría una parte) <br>
Arreglar cálculo del gradiente

> v1.2 (base subida al repositorio) VPFS con mejora en intercambio de ansatz

La base actual es la versión v1.2 pero con una pequeña mejora en la forma de cambiar los ansatz. Quedaría otra parte del código para completar esa
tarea. <br> <br>
Tareas siguientes <br>
Arreglar cálculo analítico del gradiente
U_b, resolver problema con b siendo complejo <br>

Historial de cambios post creación repositorio
============================================

> v1.3

- Refactorizado para desacoplar ansatzs de funciones y mejorar mantenimiento del código

> v1.4

- Cálculo analítico del gradiente
- Primera versión del uso de ansatz complejo

> v1.5

- Mejoras en el uso del ansatz complejo. Arreglado bug con COBYLA, mejorada homogeneidad, y añadida de vuelta la comparación en los resultados entre
  vsol y vreal

Cambios futuros
===============

- Descomposición de matrices
- Los optimizadores podrían refactorizarse también
