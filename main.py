import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#1. Carga de datos
df = pd.read_csv("heart_disease.csv")

#info
print(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")
print(df.head())
# print("Info \n")
# print(df.info)

#2 Analisis Exploratorio y Destribucion
#nulos representados por ? 
print("\n-- Conteno de nulos y Tipos")
#print(df.isnull().sum().sum(),"valores nulos detectados.")
print("Nulos  '?' detectados:", (df == '?').sum().sum())

#Balance de la variable objetivo [Target]
#target_counts=df['target'].value_counts(normalize=True) * 100



