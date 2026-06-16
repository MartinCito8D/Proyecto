import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)
CORS(app)  # Desbloquea restricciones de seguridad del navegador (CORS)

# Variables globales para persistencia en memoria local
datos_globales = None
modelo_base = None
X_test_global, y_test_global = None, None

# Columnas obligatorias bajo la rúbrica estricta del proyecto
COLUMNAS_REQUERIDAS = [
    'ingresos_mensuales', 'deuda_activa', 'historial_pagos', 'antiguedad_laboral',
    'creditos_activos', 'monto_solicitado', 'atrasos_previos', 'dependientes_economicos',
    'utilizacion_credito', 'en_riesgo'
]

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global datos_globales
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No se detectó ningún archivo en la petición'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'Nombre de archivo vacío'}), 400

    try:
        # Carga del set de datos con Pandas
        df = pd.read_csv(file)
        
        # Validar consistencia estructural de las variables obligatorias
        if not all(col in df.columns for col in COLUMNAS_REQUERIDAS):
            return jsonify({'success': False, 'message': 'Las columnas del CSV no coinciden con los requisitos del modelo'}), 400

        # 1. Limpieza de registros duplicados
        duplicados_antes = int(df.duplicated().sum())
        df = df.drop_duplicates()

        # 2. Imputación de valores faltantes/nulos usando la mediana por columna
        nulos_totales = int(df.isnull().sum().sum())
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # 3. Tratamiento de inconsistencias lógicas financieras (Dinero negativo)
        columnas_moneda = ['ingresos_mensuales', 'deuda_activa', 'monto_solicitado']
        for col in columnas_moneda:
            mediana = df[col].median()
            df[col] = df[col].apply(lambda x: mediana if x < 0 else x)
            
        # 4. Ajuste estricto de rangos porcentuales usando clip(0, 100)
        df['historial_pagos'] = df['historial_pagos'].clip(0, 100)
        df['utilizacion_credito'] = df['utilizacion_credito'].clip(0, 100)

        datos_globales = df

        return jsonify({
            'success': True,
            'registros': len(df),
            'nulos': nulos_totales,
            'duplicados': duplicados_antes
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f"Error al procesar con Pandas: {str(e)}"}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    global datos_globales, modelo_base, X_test_global, y_test_global
    if datos_globales is None:
        return jsonify({'success': False, 'message': 'Primero debes cargar y limpiar los datos'}), 400

    try:
        # 1. CAPTURA DINÁMICA DE HIPERPARÁMETROS (Frontend -> Backend)
        data_params = request.json if request.is_json else {}
        
        # Si el frontend envía parámetros los usa; de lo contrario asigna los valores base por defecto
        n_estimators = int(data_params.get('n_estimators', 100))
        max_depth = data_params.get('max_depth', 10)
        max_leaf_nodes = data_params.get('max_leaf_nodes', 50)

        # Control por si vienen campos vacíos o nulos desde JavaScript
        if max_depth is not None and max_depth != "":
            max_depth = int(max_depth)
        else:
            max_depth = None

        if max_leaf_nodes is not None and max_leaf_nodes != "":
            max_leaf_nodes = int(max_leaf_nodes)
        else:
            max_leaf_nodes = None

        # 2. PROCESAMIENTO DEL DATASET
        X = datos_globales.drop(columns=['en_riesgo'])
        y = datos_globales['en_riesgo']

        # Partición estricta de datos (Holdout 80/20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. INSTANCIACIÓN DINÁMICA DEL MODELO
        modelo_base = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            max_leaf_nodes=max_leaf_nodes, 
            random_state=42
)
        modelo_base.fit(X_train, y_train)

        X_test_global, y_test_global = X_test, y_test
        y_pred = modelo_base.predict(X_test)
        
        # 4. CÁLCULO DE MÉTRICAS MATEMÁTICAS
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Control dimensional robusto para la Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        total_clientes = len(y_test)
        pct_sin_riesgo = (int((y_pred == 0).sum()) / total_clientes) * 100
        pct_en_riesgo = (int((y_pred == 1).sum()) / total_clientes) * 100

        # 5. RETORNO DE RESULTADOS
        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            },
            'matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'distribution': {
                'sin_riesgo': f"{pct_sin_riesgo:.1f}%",
                'en_riesgo': f"{pct_en_riesgo:.1f}%"
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f"Falla en Scikit-Learn: {str(e)}"}), 500


@app.route('/api/predict', methods=['POST'])
def predict_individual():
    global modelo_base
    if modelo_base is None:
        return jsonify({'success': False, 'message': 'El modelo predictivo no ha sido inicializado'}), 400

    try:
        data = request.json
        input_data = pd.DataFrame([{
            'ingresos_mensuales': float(data['ingresos']),
            'deuda_activa': float(data['deuda']),
            'historial_pagos': float(data['historial']),
            'antiguedad_laboral': float(data['antiguedad']),
            'creditos_activos': int(data['creditos']),
            'monto_solicitado': float(data['monto']),
            'atrasos_previos': int(data['atrasos']),
            'dependientes_economicos': int(data['dependientes']),
            'utilizacion_credito': float(data['utilizacion'])
        }])

        prediccion = int(modelo_base.predict(input_data)[0])
        return jsonify({'success': True, 'prediction': prediccion})

    except Exception as e:
        return jsonify({'success': False, 'message': f"Error de dimensiones de entrada: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)