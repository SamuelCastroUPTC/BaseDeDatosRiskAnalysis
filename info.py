from IPython.display import display
import pandas as pd
import numpy as np

# ==========================================
# BLOQUE 2: Carga y limpieza de datos
# ==========================================

def cargar_datos(path="heart_disease.csv"):
    # Lee el CSV, convierte los '?' en nulos reales y fuerza todo a numérico.
    # El target 'num' se binariza: 0 = sin riesgo, 1 = con riesgo (venía como 0-4).
    df = pd.read_csv(path)
    df.replace('?', np.nan, inplace=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df.dropna(subset=['num'], inplace=True)
    return df


def resumen_datos(df):
    # Vista rápida del dataset ya limpio: tamaño, tipos de dato, estadísticas y balance de clases.
    print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas\n")
    display(df.head())
    df.info()
    display(df.describe())
    print("\nDistribución de la variable objetivo:")
    print(df['num'].value_counts())
