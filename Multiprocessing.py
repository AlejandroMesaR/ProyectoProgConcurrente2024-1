import multiprocessing as mp  # Importar multiprocessing para la programación paralela
import numpy as np  # Importar numpy para operaciones numéricas
import matplotlib.pyplot as plt  # Importar pyplot para graficar
from tqdm import tqdm  # Importar tqdm para mostrar una barra de progreso

# Función para el trabajo realizado por cada proceso en multiprocessing
def workerMultiprocessing(args):
    i, secuencia1, secuencia2 = args  # Desempaquetar los argumentos
    dotplot = np.empty(len(secuencia2), dtype=np.uint8)  # Crear un array vacío para almacenar el dotplot
    for j in range(len(secuencia2)):
        # Comparar caracteres de secuencia1 y secuencia2 para calcular el dotplot
        if secuencia1[i] == secuencia2[j]:
            if i == j:
                dotplot[j] = np.uint8(2)  # Si coinciden en la diagonal principal
            else:
                dotplot[j] = np.uint8(1)  # Si coinciden fuera de la diagonal principal
        else:
            dotplot[j] = np.uint8(0)  # Si no coinciden
    return dotplot  # Devolver el dotplot calculado

# Función para paralelizar el cálculo de dotplot utilizando multiprocessing
def paralelizarMultiprocessingDotplot(secuencia1, secuencia2, numProcesadores=mp.cpu_count()):
    tarea = [(i, secuencia1, secuencia2) for i in range(len(secuencia1))]  # Crear tareas para cada índice de secuencia1
    with mp.Pool(processes=numProcesadores) as pool:  # Crear un pool de procesos
        dotplot = []  # Lista para almacenar los resultados del dotplot
        for resultado in tqdm(pool.imap(workerMultiprocessing, tarea), total=len(tarea)):
            dotplot.append(resultado)  # Añadir cada resultado a la lista
    return np.array(dotplot, dtype=np.uint8)  # Devolver la matriz de dotplot como un array numpy de tipo uint8

# Función para graficar el análisis de tiempos, aceleraciones y eficiencias usando multiprocessing
def graficarAnalisisMultiprocessing(tiempos, aceleraciones, eficiencias, numProcesadores):
    print("Generando gráficas de Multiprocessing...")  # Mensaje de estado
    print(f"Tiempo: {tiempos} Aceleración: {aceleraciones} Eficiencia: {eficiencias} Número de procesadores: {numProcesadores}")
    
    # Configurar la figura
    plt.figure(figsize=(10, 10))
    
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
    
    # Guardar la figura como un archivo de imagen
    plt.savefig("Imagenes/Multiprocessing/graficasMultiprocessing.png")

    # Mostrar la figura en pantalla
    plt.show()
