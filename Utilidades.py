import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import cv2
import time

resultadosGenerarImagenes = []  # Lista para almacenar tiempos de generación de imágenes

# Función para leer un archivo en formato FASTA y concatenar todas las secuencias en una sola cadena
def leerArchivoFasta(nombreArchivo):
    secuencias = []
    for record in SeqIO.parse(nombreArchivo, "fasta"):
        secuencias.append(str(record.seq))
    return "".join(secuencias)

# Función para graficar un dotplot usando matplotlib
def graficarDotplot(dotplot, figNombre='dotplot.svg'):
    inicioGenerarImagenes = time.time()  # Marca el inicio del tiempo de generación de imágenes
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap="Greys", aspect="auto")
    plt.xlabel("Secuencia 1")
    plt.ylabel("Secuencia 2")
    plt.savefig(figNombre)
    resultadosGenerarImagenes.append(f"Tiempo de generación de la imagen Dotplot: {time.time() - inicioGenerarImagenes}")
    plt.show()
    

# Función para guardar una lista de resultados en un archivo de texto
def guardarResultadosArchivo(resultados, nombreArchivo="ReporteTxt/resultados.txt"):
    with open(nombreArchivo, "w") as file:
        for resultado in resultados:
            file.write(str(resultado) + "\n")

# Función para calcular la aceleración basada en tiempos de ejecución
def aceleracion(times):
    return [times[0] / i for i in times]

# Función para calcular la eficiencia basada en aceleraciones y número de hilos/procesos
def eficiencia(aceleraciones, numProcesadores):
    return [aceleraciones[i] / numProcesadores[i] for i in range(len(numProcesadores))]

# Función para aplicar un filtro de convolución a una matriz y guardar la imagen resultante
def aplicarFiltroConvolucion(matriz, pathImagen):
    inicioGenerarImagenes = time.time()  # Marca el inicio del tiempo de generación de imágenes
    
    # Definir el kernel para detectar diagonales en la matriz
    kernelDiagonales = np.array([[1, -1, -1],
                                 [-1, 1, -1],
                                 [-1, -1, 1]])
    
    # Mostrar la matriz original
    print("Matriz original")
    print(matriz)
    
    # Aplicar el filtro de convolución usando el kernel definido
    filtered_matriz = cv2.filter2D(matriz, -1, kernelDiagonales)
    print("Matriz filtrada")
    print(filtered_matriz)

    # Normalizar la matriz filtrada a un rango de 0 a 127
    matrizNormalizada = cv2.normalize(filtered_matriz, None, 0, 127, cv2.NORM_MINMAX)
    print("Matriz normalizada")
    print(matrizNormalizada)

    # Aplicar un umbral para binarizar la matriz normalizada
    valorDelUmbral = 70
    _, matrizBinaria = cv2.threshold(matrizNormalizada, valorDelUmbral, 255, cv2.THRESH_BINARY)
    print("Matriz binarizada")
    print(matrizBinaria)

    # Guardar la matriz binarizada como una imagen
    cv2.imwrite(pathImagen, matrizBinaria)
    resultadosGenerarImagenes.append(f"Tiempo de generación de la imagen filtrada: {time.time() - inicioGenerarImagenes}")
    
    guardarResultadosArchivo(resultadosGenerarImagenes, nombreArchivo="ReporteTxt/tiempoDeGeneracionDeImagenesMultip.txt")
    
    # Mostrar la imagen resultante
    cv2.imshow('Diagonales', matrizBinaria)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
