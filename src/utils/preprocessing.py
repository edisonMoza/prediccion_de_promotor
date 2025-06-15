# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_and_impute_data(df, edad_median, sat8_mode):
    """
    Realiza la limpieza inicial de columnas, imputación de 'sexo',
    imputación de 'edad', y corrección/imputación de 'sat8'.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        edad_median (float): Mediana pre-calculada de la edad para imputación.
        sat8_mode (int): Moda pre-calculada de 'sat8' para corrección/imputación.

    Returns:
        pd.DataFrame: DataFrame con limpieza e imputación aplicadas.
    """
    df_processed = df.copy()

    # Eliminación de columnas iniciales
    # Aseguramos que las columnas existan antes de intentar eliminarlas
    cols_to_drop_initial = ["hospital", "caso", "rango_edad"]
    df_processed.drop(columns=[col for col in cols_to_drop_initial if col in df_processed.columns], inplace=True)

    # Imputación de valores nulos para 'sexo'
    df_processed.loc[(df_processed["Tipo_hospitalización"] == 3) & (df_processed["sexo"].isna()), "sexo"] = 2
    df_processed.loc[(df_processed["Tipo_hospitalización"] == 1) & (df_processed["sexo"].isna()), "sexo"] = 1
    df_processed.loc[(df_processed["Tipo_hospitalización"] == 2) & (df_processed["sexo"].isna()), "sexo"] = 1

    # Imputación de valores nulos para 'edad'
    # Solo imputamos si hay nulos y si la columna existe
    if 'edad' in df_processed.columns and df_processed["edad"].isna().any():
        df_processed.loc[(df_processed["edad"].isna()), "edad"] = edad_median

    # Corrección y completando valores en 'sat8'
    if 'sat8' in df_processed.columns:
        df_processed.loc[df_processed["sat8"] > 5, "sat8"] = sat8_mode
        df_processed.loc[df_processed["sat8"].isna(), "sat8"] = sat8_mode

    return df_processed

def apply_feature_scaling(df, scaler_edad, scaler_dias_hospital):
    """
    Aplica el escalamiento a las variables 'edad' y 'N_días_hosp' usando escaladores pre-ajustados.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        scaler_edad (MinMaxScaler): Objeto MinMaxScaler ajustado para 'edad'.
        scaler_dias_hospital (MinMaxScaler): Objeto MinMaxScaler ajustado para 'N_días_hosp'.

    Returns:
        pd.DataFrame: DataFrame con las variables escaladas.
    """
    df_scaled = df.copy()
    # Asegurarse de que las columnas existen antes de intentar escalar
    if 'edad' in df_scaled.columns:
        df_scaled['edad_escalada'] = scaler_edad.transform(df_scaled[['edad']])
    if 'N_días_hosp' in df_scaled.columns:
        df_scaled['N_dias_hosp_escalada'] = scaler_dias_hospital.transform(df_scaled[['N_días_hosp']])
    return df_scaled

def reclassify_target(df):
    """
    Reclasifica la variable objetivo 'sat_general'.

    Args:
        df (pd.DataFrame): DataFrame de entrada.

    Returns:
        pd.DataFrame: DataFrame con la variable objetivo reclasificada.
    """
    df_reclassified = df.copy()
    mapeo_satisfaccion = {
        1: 0, 2: 0, 3: 0,
        4: 1, # Satisfecho
        5: 1  # Satisfecho
    }
    if 'sat_general' in df_reclassified.columns:
        df_reclassified.loc[:,'sat_general_reclasificada'] = df_reclassified["sat_general"].map(mapeo_satisfaccion)
    return df_reclassified

def select_features(df):
    """
    Selecciona las características finales para el modelo y la variable objetivo.

    Args:
        df (pd.DataFrame): DataFrame de entrada que ya ha pasado por todas las transformaciones.

    Returns:
        pd.DataFrame: DataFrame con las características seleccionadas para X.
        pd.Series: Serie con la variable objetivo y.
    """
    # Columnas originales o intermedias que no son características finales para X o Y
    cols_to_exclude_from_final_X = [
        'sexo', 'medicion', 'Procedencia', 'Tipo_hospitalización',
        'N_días_hosp', 'edad', 'sat_general'
    ]
    
    target_column = 'sat_general_reclasificada'

    # Construir la lista de columnas para X
    X_columns = [col for col in df.columns if col not in cols_to_exclude_from_final_X and col != target_column]
    
    # Asegurarse de que todas las columnas en X_columns existan en df
    X_columns = [col for col in X_columns if col in df.columns]

    X = df[X_columns]
    y = df[target_column] # Asume que esta columna ya fue creada por reclassify_target

    return X, y