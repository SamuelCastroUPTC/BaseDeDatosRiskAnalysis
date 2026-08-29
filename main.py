import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer # Librería para la imputación avanzada

# 1. Carga de datos
df = pd.read_csv("heart_disease.csv")

# Info
print(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas\n")
print(df.head())

# 2. Análisis Exploratorio y Manejo de Nulos
print("\n--- Conteo de nulos y Tipos ---")
print("Nulos '?' detectados originalmente:", (df == '?').sum().sum())

# Paso A: Reemplazar '?' por NaN (Not a Number) de Numpy
df.replace('?', np.nan, inplace=True)

# Paso B: Convertir todo a numérico. Pandas leyó algunas columnas como texto por culpa de los '?'
df = df.apply(pd.to_numeric)

# Paso C: Eliminar filas donde la variable objetivo ('num') sea nula (regla de oro)
df.dropna(subset=['num'], inplace=True)

print("Valores nulos reales (NaN) por columna antes de imputar:\n", df.isnull().sum())

# 3. Imputación Avanzada con KNN
print("\n--- Aplicando Imputación KNN ---")
# Configuramos el algoritmo para buscar a los 5 pacientes más similares
imputer = KNNImputer(n_neighbors=5, weights='distance')

# fit_transform aplica el algoritmo y devuelve una matriz de Numpy
df_imputado_matriz = imputer.fit_transform(df)

# Convertimos la matriz de vuelta a un DataFrame de Pandas
df_clean = pd.DataFrame(df_imputado_matriz, columns=df.columns)

print("Nulos totales después de la imputación:", df_clean.isnull().sum().sum())

# 4. Balance de la variable objetivo [Target]
print("\n--- Balance de la variable objetivo ('num') ---")
# En algunos datasets de cardiología, 'num' va de 0 (sano) a 4 (muy enfermo). 
# Lo binarizamos: 0 es sano, 1 o más es enfermo (1).
df_clean['num'] = df_clean['num'].apply(lambda x: 1 if x > 0 else 0)

target_counts = df_clean['num'].value_counts(normalize=True) * 100
print(target_counts.round(2).astype(str) + ' %')