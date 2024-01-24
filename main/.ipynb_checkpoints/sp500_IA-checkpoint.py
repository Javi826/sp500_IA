# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:32:01 2023

@author: jlaho
"""

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, initializers
import sys
import pandas as pd
import numpy as np
import os
import time

# Guarda el tiempo de inicio
inicio_tiempo = time.time()

# Especifica las partes de la ruta
folder = "inputs\historico"
archivo = "sp500_historico.xlsx"

# Construye la ruta absoluta
current_directory = os.getcwd()
ruta_absoluta = os.path.join(current_directory, folder, archivo)
print("Ruta absoluta:", ruta_absoluta)

ruta_salir_main = os.path.dirname(current_directory)
# Construye la ruta absoluta
ruta_absoluta = os.path.join(ruta_salir_main, folder, archivo)
#print("Ruta absoluta:", ruta_absoluta)

# Especifica la carpeta principal
carpeta_principal = 'outputs'


# Verifica si la carpeta principal existe, y si no, la crea
if not os.path.exists(os.path.join(carpeta_principal)):
    os.makedirs(os.path.join(ruta_salir_main, carpeta_principal))


# Modifica la carpeta para que se cree dentro de "outputs"
carpeta_funcional = os.path.join('outputs', 'funcional')
if not os.path.exists(carpeta_funcional):
    os.makedirs(carpeta_funcional)
    
# Verificar si se deben reiniciar los DataFrames al principio
reiniciar_dataframes = True  # Puedes ajustar esto según tus necesidades
# Reiniciar o definir el DataFrame principal
if 'df_data' in locals() and reiniciar_dataframes:
    del dataframe  # Eliminar el DataFrame existente si existe
dataframe = pd.DataFrame()

# Lee el archivo Excel omitiendo la primera fila como encabezado
df_data = pd.read_excel(ruta_absoluta, header=None, skiprows=1, names=['fecha', 'fecha_formato', 'dia_semana_dia', 'ultimo_precio', 'apertura_dia', 'maximo_dia', 'minimo_dia'])

print("Paso 1 OK: Lectura fichero")