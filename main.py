# ==========================================
# BLOQUE 1: Importación de librerías
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay
)

from info import cargar_datos, resumen_datos
from evaluate import evaluate_model
from preprocess import construir_preprocesador
from baseline import entrenar_baseline
from xgb import entrenar_xgboost, obtener_importancias

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)



    

def main():
    df = cargar_datos()
    resumen_datos(df)

    X = df.drop('num', axis=1)
    y = df['num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = construir_preprocesador(X)

    print("\nEntrenando modelo base (Regresión Logística)...")
    baseline_pipeline = entrenar_baseline(preprocessor, X_train, y_train)
    y_pred_baseline = baseline_pipeline.predict(X_test)
    y_proba_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]
    print("\nMétricas del modelo base:")
    baseline_metrics = evaluate_model(y_test, y_pred_baseline, y_proba_baseline)

    print("\nEntrenando XGBoost con búsqueda de hiperparámetros...")
    best_xgb = entrenar_xgboost(preprocessor, X_train, y_train, cv)
    y_pred_xgb = best_xgb.predict(X_test)
    y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    print("\nMétricas del modelo XGBoost:")
    xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_proba_xgb)

    df_importancias = obtener_importancias(best_xgb)

    contexto = {
        'df': df,
        'y_test': y_test,
        'y_pred_baseline': y_pred_baseline,
        'y_pred_xgb': y_pred_xgb,
        'df_importancias': df_importancias,
        'baseline_metrics': baseline_metrics,
        'xgb_metrics': xgb_metrics
    }

    menu_graficas(contexto)

if __name__ == "__main__":
    main()
    
def grafica_distribucion(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='num', data=df, hue='num', palette='viridis', legend=False)
    plt.title('Distribución de la Variable Objetivo')
    plt.show()


def grafica_correlacion(df):
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación')
    plt.show()


def grafica_matriz_confusion(y_test, y_pred, titulo, cmap):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sin riesgo', 'Con riesgo']).plot(cmap=cmap)
    plt.title(titulo)
    plt.show()


def grafica_importancias(df_importancias):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Variable', data=df_importancias.head(10), palette='viridis')
    plt.title('Top 10 Variables más Importantes según XGBoost')
    plt.xlabel('Nivel de Importancia')
    plt.ylabel('Variables Médicas')
    plt.tight_layout()
    plt.show()


def tabla_comparativa(baseline_metrics, xgb_metrics):
    # Junta las métricas de ambos modelos en un DataFrame y las dibuja como
    # tabla con matplotlib, en vez de solo imprimirlas como texto.
    df_comparativa = pd.DataFrame({
        'Regresión Logística': baseline_metrics,
        'XGBoost': xgb_metrics
    }).round(4)

    fig, ax = plt.subplots(figsize=(7, 2 + 0.4 * len(df_comparativa)))
    ax.axis('off')

    tabla = ax.table(
        cellText=df_comparativa.values,
        rowLabels=df_comparativa.index,
        colLabels=df_comparativa.columns,
        cellLoc='center',
        rowLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1, 1.8)

    # Encabezados en negrita con fondo oscuro
    for (fila, col), celda in tabla.get_celld().items():
        if fila == 0:
            celda.set_text_props(weight='bold', color='white')
            celda.set_facecolor('#404040')
        elif col == -1:
            celda.set_text_props(weight='bold')
            celda.set_facecolor('#f0f0f0')

    plt.title('Comparación de métricas: Baseline vs XGBoost', fontsize=13, pad=20)
    plt.tight_layout()
    plt.show()


# ==========================================
# BLOQUE 3: Menú interactivo
# ==========================================

def menu_graficas(contexto):
    opciones = {
        '1': ('Distribución de la variable objetivo', lambda: grafica_distribucion(contexto['df'])),
        '2': ('Matriz de correlación', lambda: grafica_correlacion(contexto['df'])),
        '3': ('Matriz de confusión - Regresión Logística', lambda: grafica_matriz_confusion(
            contexto['y_test'], contexto['y_pred_baseline'], 'Matriz de Confusión - Modelo Base (LogReg)', 'Blues')),
        '4': ('Matriz de confusión - XGBoost', lambda: grafica_matriz_confusion(
            contexto['y_test'], contexto['y_pred_xgb'], 'Matriz de Confusión - XGBoost', 'Greens')),
        '5': ('Importancia de variables (XGBoost)', lambda: grafica_importancias(contexto['df_importancias'])),
        '6': ('Comparar métricas: Baseline vs XGBoost', lambda: tabla_comparativa(
            contexto['baseline_metrics'], contexto['xgb_metrics'])),
        '0': ('Salir', None)
    }

    while True:
        print("\n--- Menú de gráficas ---")
        for clave, (nombre, _) in opciones.items():
            print(f"{clave}. {nombre}")

        eleccion = input("Elige una opción: ").strip()
        if eleccion == '0':
            break
        elif eleccion in opciones:
            opciones[eleccion][1]()
        else:
            print("Opción no válida, intenta de nuevo.")

