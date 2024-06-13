import time
from mpi4py import MPI
import argparse
import numpy as np
from MPI import *  # Importa funciones específicas para MPI
from Multiprocessing import *  # Importa funciones específicas para multiprocessing
from Secuencial import *  # Importa funciones específicas para ejecución secuencial
from Utilidades import *  # Importa utilidades adicionales

def main():
    
    tiempoInicioNoParalelo = time.time()  # Marca el inicio del tiempo de ejecución del bloque no paralelo
    
    # Inicialización de MPI
    comm = MPI.COMM_WORLD  # Obtiene el comunicador global
    rank = comm.Get_rank()  # Obtiene el rango del proceso actual
    size = comm.Get_size()  # Obtiene el número total de procesos

    # Configuración de los argumentos del script
    parser = argparse.ArgumentParser()

    # Agrega argumentos al parser para especificar archivos, modos de ejecución y otros parámetros
    parser.add_argument('--file1', dest='archivo1', type=str, default=None, help='Archivo 1 de secuencia en formato FASTA')
    parser.add_argument('--file2', dest='archivo2', type=str, default=None, help='Archivo 2 de secuencia en formato FASTA')
    parser.add_argument('--maxLen', dest='maxLen', type=int, default=10000, help='Max tamaño de las secuencias a comparar')
    parser.add_argument('--sequential', action='store_true', help='Ejecutar en modo secuencial')
    parser.add_argument('--multiprocessing', action='store_true', help='Ejecutar utilizando multiprocessing')
    parser.add_argument('--mpi', action='store_true', help='Ejecutar utilizando mpi4py')
    parser.add_argument('--num_processes', dest='num_procesadores', type=int, nargs='+', default=[1, 2, 4, 8, 16], help='Número de procesos para la opción MPI')
    args = parser.parse_args()

    # Solo el proceso con rank 0 realiza la carga y preparación de datos
    
    cargaArchivoInicio = time.time()  # Marca el inicio del tiempo de carga de archivos
    archivoPath1 = args.archivo1  # Ruta del archivo 1
    archivoPath2 = args.archivo2  # Ruta del archivo 2

    numProcesadoresArray = args.num_procesadores  # Lista de números de procesadores para pruebas

    try:
        # Leer los archivos FASTA
        secuenciaTotal1 = leerArchivoFasta(archivoPath1)
        secuenciaTotal2 = leerArchivoFasta(archivoPath2)
    except FileNotFoundError as e:
        print("Archivo no encontrado, verifique la ruta")
        exit(1)

    # Reducir tamaño de las secuencias para manejar el problema de memoria
    maxLen = args.maxLen  # Máximo tamaño permitido para las secuencias
    Secuencia1 = secuenciaTotal1[:maxLen]  # Recorta la secuencia 1
    Secuencia2 = secuenciaTotal2[:maxLen]  # Recorta la secuencia 2
    cargaArchivoFinal = time.time()  # Marca el final del tiempo de carga de archivos

    # Guardar el tiempo de carga de los archivos
    guardarResultadosArchivo([f"Tiempo de carga de los archivos: {cargaArchivoFinal - cargaArchivoInicio}"], 
                                nombreArchivo="ReporteTxt/tiempoDeCargaArchivos.txt")

    # Inicializar el dotplot y las listas para guardar resultados
    dotplot = np.empty([len(Secuencia1), len(Secuencia2)], dtype=np.uint8)  # Matriz vacía para el dotplot
    resultadosPrint = []  # Lista para almacenar resultados de multiprocessing
    resultadosPrintMPI = []  # Lista para almacenar resultados de MPI
    tiemposMultiprocessing = []  # Lista para almacenar tiempos de multiprocessing
    tiemposMPI = []  # Lista para almacenar tiempos de MPI

    
    tiempoFinalNoParalelo = time.time() - tiempoInicioNoParalelo  # Calcula el tiempo total de ejecución del bloque no paralelo
    
    # Ejecutar en modo multiprocessing si se especifica en los argumentos
    if args.multiprocessing:
        for cantidadProcesadores in numProcesadoresArray:
            tiempoInicioPacial = time.time()  # Marca el inicio del tiempo de procesamiento
            # Ejecuta el dotplot en paralelo usando multiprocessing
            dotplotMultiprocessing = np.array(paralelizarMultiprocessingDotplot(Secuencia1, Secuencia2, 
                                                                                numProcesadores=cantidadProcesadores), dtype=np.uint8)
            tiempoTotalPacial = time.time() - tiempoInicioPacial  # Calcula el tiempo total de ejecución
            tiemposMultiprocessing.append(tiempoTotalPacial)
            resultadosPrint.append(f"Tiempo de ejecución parcial con {cantidadProcesadores} procesadores: {tiempoTotalPacial}")

            # Graficar y filtrar el dotplot
            graficarDotplot(dotplotMultiprocessing[:2000, :2000], figNombre=f"Imagenes/Multiprocessing/dotplot_{cantidadProcesadores}_procesadores.png")
            
            pathImagen = f'Imagenes/Filtradas/dotplotFiltrado_{cantidadProcesadores}_procesadores.png'
            aplicarFiltroConvolucion(dotplotMultiprocessing[:2000, :2000], pathImagen)
        
        # Calcular aceleración y eficiencia
        aceleraciones = aceleracion(tiemposMultiprocessing)
        for i in range(len(aceleraciones)):
            resultadosPrint.append(f"Aceleración con {numProcesadoresArray[i]} procesadores: {aceleraciones[i]}")
        
        eficiencias = eficiencia(aceleraciones, numProcesadoresArray)
        for i in range(len(eficiencias)):
            resultadosPrint.append(f"Eficiencia con {numProcesadoresArray[i]} procesadores: {eficiencias[i]}")
        
        # Guardar resultados y graficar análisis
        graficarAnalisisMultiprocessing(tiemposMultiprocessing, aceleraciones, eficiencias, numProcesadoresArray)
        graficarDotplot(dotplotMultiprocessing[:2000, :2000], figNombre='Imagenes/Multiprocessing/dotplotMultiprocessing.png')
        pathImagen = 'Imagenes/Filtradas/dotplotFiltradoMultiprocessing.png'  
        aplicarFiltroConvolucion(dotplotMultiprocessing[:2000, :2000], pathImagen)
        
        # Calcular tiempo de ejecución en bloque
        for i in range(len(numProcesadoresArray)):
            resultadosPrint.append(f"Tiempo de ejecución en bloque con {numProcesadoresArray[i]} procesadores: {tiemposMultiprocessing[i]+tiempoFinalNoParalelo}")
        
        guardarResultadosArchivo(resultadosPrint, nombreArchivo="ReporteTxt/resultadosMultiprocessing.txt")

    # Ejecutar en modo MPI si se especifica en los argumentos
    if args.mpi:
        if rank == 0:
            for cantidadProcesadores in numProcesadoresArray:
                tiempoInicioPacial = time.time()  # Marca el inicio del tiempo de procesamiento
                dotplot = paralelizarMpiDotplot(Secuencia1, Secuencia2)  # Ejecuta el dotplot en paralelo usando MPI
                tiempoTotalPacial = time.time() - tiempoInicioPacial  # Calcula el tiempo total de ejecución
                tiemposMPI.append(tiempoTotalPacial)
                resultadosPrintMPI.append(f"Tiempo de ejecución con {cantidadProcesadores} procesadores: {tiempoTotalPacial}")

                # Graficar y filtrar el dotplot
                graficarDotplot(dotplot[:2000, :2000], figNombre=f"Imagenes/MPI/dotplot_{cantidadProcesadores}_procesadores.png")
                
                pathImagen = f'Imagenes/Filtradas/dotplotFiltradoMPI_{cantidadProcesadores}_procesadores.png'
                aplicarFiltroConvolucion(dotplot[:2000, :2000], pathImagen)
          
            # Calcular aceleración y eficiencia
            aceleraciones = aceleracion(tiemposMPI)
            for i in range(len(aceleraciones)):
                resultadosPrintMPI.append(f"Aceleración con {numProcesadoresArray[i]} procesadores: {aceleraciones[i]}")
            
            eficiencias = eficiencia(aceleraciones, numProcesadoresArray)
            for i in range(len(eficiencias)):
                resultadosPrintMPI.append(f"Eficiencia con {numProcesadoresArray[i]} procesadores: {eficiencias[i]}")

            # Guardar resultados y graficar análisis
            graficarAnalisisMPI(tiemposMPI, aceleraciones, eficiencias, numProcesadoresArray)
            graficarDotplot(dotplot[:2000, :2000], figNombre='Imagenes/MPI/dotplotMPI.png')
            pathImagen = 'Imagenes/Filtradas/dotplotFiltradoMPI.png'  
            aplicarFiltroConvolucion(dotplot[:2000, :2000], pathImagen)
            
            # Calcular tiempo de ejecución en bloque
            for i in range(len(numProcesadoresArray)):
                resultadosPrintMPI.append(f"Tiempo de ejecución en bloque con {numProcesadoresArray[i]} procesadores: {tiemposMPI[i]+tiempoFinalNoParalelo}")
            
            guardarResultadosArchivo(resultadosPrintMPI, nombreArchivo="ReporteTxt/ResultadosMPI.txt")

    # Ejecutar en modo secuencial si se especifica en los argumentos
    if args.sequential:
        inicioSecuencial = time.time()  # Marca el inicio del tiempo de procesamiento secuencial
        dotplotSequential = sequentialDotplot(Secuencia1, Secuencia2)  # Ejecuta el dotplot de forma secuencial
        tiempoTotalPacial = time.time() - inicioSecuencial  # Calcula el tiempo total de ejecución
        resultadosPrint.append(f"Tiempo de ejecución secuencial: {tiempoTotalPacial}")

        # Graficar y filtrar el dotplot
        graficarDotplot(dotplotSequential[:5000, :5000], figNombre="Imagenes/Secuencial/dotplotSecuencial.png")
        
        pathImagen = 'Imagenes/Filtradas/dotplotFiltradaSequential.png'
        aplicarFiltroConvolucion(dotplotSequential[:5000, :5000], pathImagen)
        
        # Calcular tiempo de ejecución en bloque
        resultadosPrint.append(f"Tiempo de ejecución en bloque secuencial: {tiempoTotalPacial+tiempoFinalNoParalelo}")
        
        guardarResultadosArchivo(resultadosPrint, nombreArchivo="ReporteTxt/resultadoSequential.txt")

if __name__ == '__main__':
    main()
