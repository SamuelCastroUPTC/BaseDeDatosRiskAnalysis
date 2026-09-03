import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


# ==========================================
# BLOQUE 6: Modelo avanzado XGBoost
# ==========================================
# Busca la mejor combinación de hiperparámetros con GridSearchCV usando
# la misma validación cruzada del modelo base, para comparar en igualdad
# de condiciones.

def entrenar_xgboost(preprocessor, X_train, y_train, cv):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# ==========================================
# BLOQUE 7: Importancia de variables
# ==========================================
# Extrae qué tanto pesó cada variable en las decisiones del modelo XGBoost
# ya entrenado, para entender qué factores médicos influyen más.

def obtener_importancias(modelo_pipeline):
    modelo = modelo_pipeline.named_steps['classifier']
    feature_names = modelo_pipeline.named_steps['preprocessor'].get_feature_names_out()
    df_importancias = pd.DataFrame({
        'Variable': feature_names,
        'Importancia': modelo.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    df_importancias['Variable'] = df_importancias['Variable'].str.replace('num__', '').str.replace('cat__', '')
    return df_importancias

