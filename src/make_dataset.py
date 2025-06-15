# src/make_dataset.py

import pandas as pd
import numpy as np
import os
import pickle # Necesario para guardar los escaladores y valores de imputación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # Para ajustar los escaladores
from utils.preprocessing import clean_and_impute_data, apply_feature_scaling, reclassify_target, select_features

def run_make_dataset(raw_data_path, processed_data_dir, models_dir):
    """
    Carga los datos brutos, realiza el preprocesamiento, la ingeniería de características,
    la división en conjuntos de entrenamiento y prueba, y guarda los datos procesados
    y los objetos/valores de preprocesamiento.

    Args:
        raw_data_path (str): Ruta al archivo de datos brutos.
        processed_data_dir (str): Directorio donde se guardarán los datos procesados.
        models_dir (str): Directorio donde se guardarán los objetos de escalado e imputación.
    """
    print("Iniciando la preparación de los datos...")

    # Asegurarse de que los directorios de salida existan
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Cargar datos brutos
    try:
        raw_data = pd.read_excel(raw_data_path, sheet_name="SAT HOSP_FEATURE")
        print(f"'{os.path.basename(raw_data_path)}' cargado correctamente.")
    except FileNotFoundError:
        print(f"Error: Archivo '{raw_data_path}' no encontrado. Asegúrate de que la ruta sea correcta.")
        return
    except Exception as e:
        print(f"Error al cargar el archivo Excel: {e}")
        return

    df = raw_data.copy()

    # 2. Calcular y guardar valores de imputación (mediana y moda)
    # Se calculan AQUI una vez sobre los datos brutos, y se guardan.
    # Así predict.py los usará consistentemente.
    edad_median = df["edad"].median()
    sat8_moda = df["sat8"].mode().iloc[0]

    with open(os.path.join(models_dir, 'imputation_values.pkl'), 'wb') as f:
        pickle.dump({'edad_median': edad_median, 'sat8_mode': sat8_moda}, f)
    print(f"Valores de imputación (mediana edad: {edad_median}, moda sat8: {sat8_moda}) guardados.")

    # 3. Aplicar limpieza e imputación inicial usando la función de preprocessing.py
    df_cleaned = clean_and_impute_data(df, edad_median, sat8_moda)
    print("Limpieza e imputación de datos iniciales aplicadas.")

    # 4. Ajustar y guardar los escaladores (MinMaxScaler)
    # Se ajustan AQUI sobre los datos limpios, y se guardan.
    scaler_edad = MinMaxScaler()
    df_cleaned['edad_escalada_temp'] = scaler_edad.fit_transform(df_cleaned[['edad']]) # fit_transform solo para ajustar y obtener datos escalados temporalmente
    
    scaler_dias_hospital = MinMaxScaler()
    df_cleaned['N_dias_hosp_escalada_temp'] = scaler_dias_hospital.fit_transform(df_cleaned[['N_días_hosp']]) # fit_transform solo para ajustar y obtener datos escalados temporalmente

    # Guardar los escaladores ajustados para su uso futuro (predicción)
    with open(os.path.join(models_dir, 'scaler_edad.pkl'), 'wb') as f:
        pickle.dump(scaler_edad, f)
    with open(os.path.join(models_dir, 'scaler_dias_hospital.pkl'), 'wb') as f:
        pickle.dump(scaler_dias_hospital, f)
    print("Escaladores 'edad' y 'N_días_hosp' ajustados y guardados.")

    # 5. Aplicar escalamiento a los datos usando los escaladores ajustados
    # Usamos apply_feature_scaling, que solo hace .transform()
    df_scaled = apply_feature_scaling(df_cleaned, scaler_edad, scaler_dias_hospital)
    # Eliminamos las columnas temporales si se crearon
    df_scaled.drop(columns=[col for col in ['edad_escalada_temp', 'N_dias_hosp_escalada_temp'] if col in df_scaled.columns], inplace=True)
    print("Escalamiento de variables aplicado.")
    
    # 6. Reclasificación de la variable objetivo 'sat_general'
    df_reclassified = reclassify_target(df_scaled)
    print("Variable 'sat_general' reclasificada a 'sat_general_reclasificada'.")

    # 7. Selección final de variables y división en X e y
    X, y = select_features(df_reclassified)
    print(f"Variables seleccionadas para X: {X.columns.tolist()}")
    print("Variable objetivo (y) definida.")
    
    # 8. Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Datos divididos en entrenamiento (80%) y prueba (20%).")
    print(f"Tamaño de X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Tamaño de X_test: {X_test.shape}, y_test: {y_test.shape}")

    # 9. Guardar los conjuntos de datos procesados
    X_train.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)
    print("Datos de entrenamiento y prueba guardados en la carpeta 'processed'.")

    print("Preparación de datos finalizada exitosamente.")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, '..')
    
    raw_data_file = os.path.join(project_root, 'data', 'raw', 'DATA DE SATISFACCIÓN HOSPITAL APP_SESIÓN 03.xlsx')
    processed_data_output_dir = os.path.join(project_root, 'data', 'processed')
    models_output_dir = os.path.join(project_root, 'models')

    run_make_dataset(raw_data_file, processed_data_output_dir, models_output_dir)