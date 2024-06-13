from tqdm import tqdm  # Importar la barra de progreso
import numpy as np  # Importar numpy para operaciones numéricas

# Función para calcular el dotplot de manera secuencial
def sequentialDotplot(sequence1, sequence2):
    # Crear una matriz vacía para almacenar el dotplot
    dotplot = np.empty((len(sequence1), len(sequence2)), dtype=np.uint8)
    
    # Iterar sobre los índices de las secuencias y calcular el dotplot
    for i in tqdm(range(len(sequence1))):  # Usar tqdm para mostrar una barra de progreso
        for j in range(len(sequence2)):
            # Comparar caracteres de sequence1 y sequence2 para calcular el dotplot
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot[i, j] = np.uint8(2)  # Si coinciden en la diagonal principal
                else:
                    dotplot[i, j] = np.uint8(1)  # Si coinciden fuera de la diagonal principal
            else:
                dotplot[i, j] = np.uint8(0)  # Si no coinciden

    # Imprimir mensaje de finalización y mostrar la matriz dotplot
    print("Dotplot secuencial terminado")
    print(dotplot)

    # Devolver la matriz dotplot calculada
    return dotplot
