from tqdm import tqdm  # Importar la barra de progreso
import numpy as np  # Importar numpy para operaciones numéricas
import time  # Importar time para medir tiempos de ejecución
import matplotlib.pyplot as plt  # Importar pyplot para graficar
from mpi4py import MPI  # Importar mpi4py para MPI

# Función para paralelizar el cálculo de dotplot utilizando MPI
def paralelizarMpiDotplot(secuencia1, secuencia2):
    comm = MPI.COMM_WORLD  # Obtener el comunicador MPI
    rank = comm.Get_rank()  # Obtener el rango (rank) del proceso actual
    size = comm.Get_size()  # Obtener el tamaño (número de procesos) del comunicador

    # Dividir el índice de secuencia1 en partes iguales entre los procesos
    chunks = np.array_split(range(len(secuencia1)), size)

    # Crear una matriz vacía para almacenar el dotplot local de cada proceso
    dotplot = np.empty([len(chunks[rank]), len(secuencia2)], dtype=np.uint8)

    # Iterar sobre los índices asignados al proceso actual
    for i in tqdm(range(len(chunks[rank]))):  # Usar tqdm para mostrar una barra de progreso
        for j in range(len(secuencia2)):
            # Comparar caracteres de secuencia1 y secuencia2 para calcular el dotplot
            if secuencia1[chunks[rank][i]] == secuencia2[j]:
                if i == j:
                    dotplot[i, j] = np.uint8(2)  # Si coinciden en la diagonal principal
                else:
                    dotplot[i, j] = np.uint8(1)  # Si coinciden fuera de la diagonal principal
            else:
                dotplot[i, j] = np.uint8(0)  # Si no coinciden

    # Recopilar todos los dotplots locales en el proceso 0
    dotplot = comm.gather(dotplot, root=0)

    # Proceso 0: combinar los dotplots y devolver el resultado final
    if rank == 0:
        merged_data = np.vstack(dotplot)  # Combinar los dotplots recopilados
        return merged_data

# Función para graficar análisis de tiempos, aceleraciones y eficiencias usando MPI
def graficarAnalisisMPI(tiempos, aceleraciones, eficiencias, numProcesadores):
    plt.figure(figsize=(10, 10))  # Configurar el tamaño de la figura

    # Subtrama 1: gráfico de tiempos vs número de procesadores
    plt.subplot(1, 2, 1)
    plt.plot(numProcesadores, tiempos)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo")
    
    # Subtrama 2: gráfico de aceleraciones y eficiencias vs número de procesadores
    plt.subplot(1, 2, 2)
    plt.plot(numProcesadores, aceleraciones)
    plt.plot(numProcesadores, eficiencias)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Aceleración y Eficiencia")
    plt.legend(["Aceleración", "Eficiencia"])
    
    # Guardar la figura como archivo de imagen
    plt.savefig("Imagenes/MPI/graficasMPI.png")

    # Mostrar la figura en pantalla
    plt.show()
