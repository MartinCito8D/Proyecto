import pandas as pd
import numpy as np

def limpiar_dataset(filepath):
    """
    Carga el CSV, imputa nulos, remueve duplicados y sanitiza inconsistencias financieras.
    """
    df = pd.read_csv(filepath)
    total_cargados = len(df)
    
    # Contar nulos antes de limpiar
    conteo_nulos_inicial = int(df.isnull().sum().sum())
    
    # Imputar valores nulos con la mediana de cada columna numérica
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
            
    # Contar y eliminar duplicados
    duplicados_antes = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    
    # Sanitización de inconsistencias lógicas (Evitar dinero negativo)
    columnas_dinero = ['ingresos_mensuales', 'deuda_activa', 'monto_solicitado']
    for col in columnas_dinero:
        if col in df.columns:
            mediana = df[col].median()
            df[col] = df[col].apply(lambda x: mediana if x <= 0 else x)
            
    # Validar rangos de porcentaje (0 - 100)
    columnas_porcentaje = ['historial_pagos', 'utilizacion_credito']
    for col in columnas_porcentaje:
        if col in df.columns:
            df[col] = df[col].clip(0, 100)
            
    return df, total_cargados, conteo_nulos_inicial, duplicados_antes