# ==========================================
# BLOQUE 1: Importación de librerías
# ==========================================

from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

# Configuración visual para las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# BLOQUE 2: Carga, Limpieza y Exploración de Datos (EDA)
# ==========================================

# 1. Carga de datos
df = pd.read_csv("heart_disease.csv")

# Reemplazar '?' por nulos reales (NaN)
df.replace('?', np.nan, inplace=True)

# Forzar conversión a numérico 
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Binarizar el target (0: Sin riesgo, >0: Con riesgo)
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.dropna(subset=['num'], inplace=True) # Eliminar filas sin target

# 2. Exploración de Datos
print("="*50)
print("INFORMACIÓN DEL DATASET (Limpio)")
print("="*50)
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas\n")

print("Primeras 5 filas:")
display(df.head())

print("\nInformación general:")
df.info()

print("\nEstadísticas descriptivas:")
display(df.describe())

print("\nDistribución de la variable objetivo:")
print(df['num'].value_counts())
print(f"Proporción:\n{df['num'].value_counts(normalize=True).round(2)*100}")

# 4. Gráficas
plt.figure(figsize=(6,4))
sns.countplot(x='num', data=df, hue='num', palette='viridis', legend=False)
plt.title('Distribución de la Variable Objetivo')
plt.show()

plt.figure(figsize=(12,8))
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# ==========================================
# BLOQUE 3: Función de Evaluación
# ==========================================
# Definimos una función para no repetir código al evaluar cada modelo
def evaluate_model(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
            'F1': f1, 'ROC-AUC': roc_auc}
    
# ==========================================
# BLOQUE 4: División de los Datos (Train/Test Split)
# ==========================================
# 1. Separar características (X) y la variable objetivo (y)
X = df.drop('num', axis=1)
y = df['num']

# 2. Dividir en conjunto de Entrenamiento (80%) y Prueba (20%)
# stratify=y asegura que se mantenga la proporción original de sanos/enfermos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Configurar el protocolo de Validación Cruzada (5 pliegues)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Datos divididos correctamente.")
print(f"Entrenamiento: {X_train.shape[0]} pacientes.")
print(f"Prueba: {X_test.shape[0]} pacientes.")


# ==========================================
# BLOQUE 3: Preprocesamiento (Pipelines)
# ==========================================
# 1. Identificar automáticamente qué columnas son números y cuáles son texto/categorías
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 2. Crear las "tuberías" (pipelines) de transformación para los números
#    - Rellena nulos con la mediana
#    - Escala los datos para que todos tengan el mismo peso
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3. Crear las "tuberías" para las variables categóricas
#    - Rellena nulos con el valor más frecuente
#    - Convierte texto a números (One-Hot Encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Unir todo en un solo "preprocesador" maestro
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Preprocesador configurado correctamente.")
# Opcional: ver cuántos nulos había antes de que el preprocesador haga su magia
print("\nCantidad de nulos en el dataset original:")
print(df.isnull().sum())

# ==========================================
# BLOQUE 5: Modelo Base (Regresión Logística)
# ==========================================
print("Entrenando el Modelo Base (Regresión Logística)...\n")

# 1. Crear el pipeline que junta el preprocesador y el modelo de regresión
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Entrenar el modelo con los datos de entrenamiento
baseline_pipeline.fit(X_train, y_train)

# 3. Hacer predicciones con los datos de prueba
y_pred_baseline = baseline_pipeline.predict(X_test)
y_proba_baseline = baseline_pipeline.predict_proba(X_test)[:, 1]

# 4. Evaluar las métricas usando la función que creamos en el Bloque 5
print("Métricas del modelo Base:")
baseline_metrics = evaluate_model(y_test, y_pred_baseline, y_proba_baseline)

# 5. Evaluar con Validación Cruzada (los 5 exámenes sorpresa)
cv_scores_baseline = cross_val_score(baseline_pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"\nROC-AUC promedio en Validación Cruzada (CV): {cv_scores_baseline.mean():.4f}")

# 6. Generar y mostrar la Matriz de Confusión
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=['Sin riesgo', 'Con riesgo']).plot(cmap='Blues')
plt.title("Matriz de Confusión - Modelo Base (LogReg)")
plt.show()

# ==========================================
# BLOQUE 6: Modelo Avanzado XGBoost
# ==========================================

# 1. Crear el Pipeline integrando el preprocesador y el modelo
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
])

# 2. Definir los hiperparámetros a probar (GridSearchCV)
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# 3. Entrenar y buscar la mejor combinación
print("Iniciando búsqueda de hiperparámetros para XGBoost...")
grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# 4. Extraer el mejor modelo
best_xgb = grid_search.best_estimator_

# 5. Hacer predicciones con los datos de prueba
y_pred_xgb = best_xgb.predict(X_test)
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]

# 6. Evaluar el modelo
print("\nMétricas del modelo XGBoost Optimizado:")
xgb_metrics = evaluate_model(y_test, y_pred_xgb, y_proba_xgb)

# 7. Generar y mostrar la Matriz de Confusión
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Sin riesgo', 'Con riesgo']).plot(cmap='Greens')
plt.title("Matriz de Confusión - XGBoost")
plt.show()

# ==========================================
# BLOQUE 8: Importancia de Variables
# ==========================================

# 1. Extraer el modelo XGBoost y los nombres de las variables del pipeline
modelo_xgb = best_xgb.named_steps['classifier']
feature_names = best_xgb.named_steps['preprocessor'].get_feature_names_out()

# 2. Extraer los valores de importancia
importancias = modelo_xgb.feature_importances_

# 3. Crear una tabla para ordenar las variables de mayor a menor importancia
df_importancias = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Limpiar los nombres de las variables para que se vean mejor en la gráfica
df_importancias['Variable'] = df_importancias['Variable'].str.replace('num__', '').str.replace('cat__', '')

# 4. Graficar el Top 10 de variables más importantes
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=df_importancias.head(10), palette='viridis')
plt.title('Top 10 Variables más Importantes según XGBoost', fontsize=14)
plt.xlabel('Nivel de Importancia (Contribución al modelo)', fontsize=12)
plt.ylabel('Variables Médicas', fontsize=12)
plt.tight_layout()
plt.show()