from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ==========================================
# BLOQUE 5: Modelo base (Regresión Logística)
# ==========================================

def entrenar_baseline(preprocessor, X_train, y_train):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
