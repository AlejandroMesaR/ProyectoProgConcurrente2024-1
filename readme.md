# Proyecto Final:

## Análisis de Rendimiento de Dotplot

## Secuencial vs Paralelización

El objetivo de este proyecto es implementar y analizar el rendimiento de tres formas de realizar un dotplot, una técnica comúnmente utilizada en bioinformática para comparar secuencias de ADN o proteínas.

Se implemento una versión secuencial, una versión paralela utilizando la biblioteca multiprocessing de Python, una versión paralela utilizando mpi4py y una versión usando pyCuda. Compara el rendimiento de estas tres implementaciones utilizando varias métricas.

### Prerequisitos

El proyecto se desarrolló utilizando Python 3.11.5 y cuenta con soporte para computación paralela mediante las bibliotecas multiprocessing y mpi4py. Requiere que se especifiquen parámetros de entrada, como las secuencias de referencia y consulta en formato fasta(.fna), los cuales deben ser declarados en la línea de comandos al momento de la ejecución para calcular el dot-plot, tambien tener en cuenta el numeor de procesadores disponibles y la memoria RAM del PC que se va a ejecutar y tambien tener disponibilidad para usar Colab para ejecutar la implementacion en PyCuda.

### Instalacion

Tener instalado python y tener las siguientes librerias necesarias, en caso de no tenerlas instalarlas con el siguiente comando:

```
pip install numpy
pip install matplotlib
pip install mpi4py
pip install biopython
pip install opencv-python
pip install tqdm==2.2.3
```

```
Link del repositorio para: https://github.com/AlejandroMesaR/ProyectoProgConcurrente2024-1.git
```

### Ejecución

Para ejecutar el programa secuencial, ejecute el siguiente comando:

```
python Main.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --sequential

o 

python Main.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --maxLen=20000 --sequential
```

Para ejecutar multiprocessing, ejecute el siguiente comando:

```
python Main.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --multiprocessing

o 

python Main.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --maxLen=20000 --multiprocessing
```

Para ejecutar mpi4py, ejecute el siguiente comando:

```
python Main.py --num_processes 1 2 4 8 --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --mpi

o 

python Main.py --num_processes 1 2 4 8 --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --maxLen=20000 --mpi
```
