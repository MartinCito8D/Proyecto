import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'credit_model.pkl')

def entrenar_modelo_rf(df, n_estimators=100, max_depth=10, max_leaf_nodes=50):
    """
    Entrena un Random Forest basado en las variables del enunciado y calcula métricas.
    """
    features = [
        'ingresos_mensuales', 'deuda_activa', 'historial_pagos', 
        'antiguedad_laboral', 'creditos_activos', 'monto_solicitado', 
        'atrasos_previos', 'dependientes_economicos', 'utilizacion_credito'
    ]
    target = 'en_riesgo'
    
    if not all(col in df.columns for col in features + [target]):
        raise ValueError("El dataset no cuenta con todas las columnas obligatorias.")
        
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth else None,
        max_leaf_nodes=int(max_leaf_nodes) if max_leaf_nodes else None,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcular métricas básicas
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 1),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0) * 100, 1)
    }
    
    # Distribución para las gráficas/tarjetas
    total_preds = len(y_pred)
    en_riesgo_count = int((y_pred == 1).sum())
    metrics['pct_sin_riesgo'] = round(((total_preds - en_riesgo_count) / total_preds) * 100)
    metrics['pct_en_riesgo'] = round((en_riesgo_count / total_preds) * 100)
    
    # Matriz de Confusión
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['matrix'] = {
        'tn': int(tn), 'fp': int(fp),
        'fn': int(fn), 'tp': int(tp)
    }
    
    joblib.dump(model, MODEL_PATH)
    return metrics

def ejecutar_prediccion_individual(datos_cliente):
    """
    Carga el archivo .pkl del modelo entrenado y clasifica a un nuevo aplicante.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No se ha entrenado ningún modelo aún.")
        
    model = joblib.load(MODEL_PATH)
    
    features_ordenados = [[
        float(datos_cliente['ingresos_mensuales']), float(datos_cliente['deuda_activa']),
        float(datos_cliente['historial_pagos']), float(datos_cliente['antiguedad_laboral']),
        int(datos_cliente['creditos_activos']), float(datos_cliente['monto_solicitado']),
        int(datos_cliente['atrasos_previos']), int(datos_cliente['dependientes_economicos']),
        float(datos_cliente['utilizacion_credito'])
    ]]
    
    prediccion = model.predict(features_ordenados)[0]
    return int(prediccion)