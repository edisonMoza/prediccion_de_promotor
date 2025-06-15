# src/predict.py

import pandas as pd
import pickle
import os
from utils.preprocessing import clean_and_impute_data, apply_feature_scaling, select_features, reclassify_target

def run_predict_model(new_raw_data_path, models_dir, predictions_dir):
    """
    Carga nuevos datos crudos, los preprocesa usando los artefactos guardados,
    carga el modelo entrenado, realiza predicciones y guarda los resultados.

    Args:
        new_raw_data_path (str): Ruta al archivo de nuevos datos crudos para predicción.
        models_dir (str): Directorio donde se encuentran el modelo entrenado y los artefactos de preprocesamiento.
        predictions_dir (str): Directorio donde se guardarán los resultados de las predicciones.
    """
    print("Iniciando el proceso de predicción...")

    # Asegurarse de que el directorio de salida de predictions exista
    os.makedirs(predictions_dir, exist_ok=True)

    # 1. Cargar el modelo entrenado
    model_path = os.path.join(models_dir, 'best_model.pkl')
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Modelo cargado correctamente desde '{model_path}'.")
    except FileNotFoundError:
        print(f"Error: Modelo no encontrado en '{model_path}'. "
              "Asegúrate de haber ejecutado 'train.py' primero.")
        return
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 2. Cargar los artefactos de preprocesamiento (valores de imputación y escaladores)
    imputation_values_path = os.path.join(models_dir, 'imputation_values.pkl')
    scaler_edad_path = os.path.join(models_dir, 'scaler_edad.pkl')
    scaler_dias_hospital_path = os.path.join(models_dir, 'scaler_dias_hospital.pkl')

    try:
        with open(imputation_values_path, 'rb') as f:
            imputation_values = pickle.load(f)
        edad_median = imputation_values['edad_median']
        sat8_mode = imputation_values['sat8_mode']
        
        with open(scaler_edad_path, 'rb') as f:
            scaler_edad = pickle.load(f)
        
        with open(scaler_dias_hospital_path, 'rb') as f:
            scaler_dias_hospital = pickle.load(f)
        print("Artefactos de preprocesamiento (valores de imputación, escaladores) cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: Uno o más artefactos de preprocesamiento no encontrados en '{models_dir}'. "
              "Asegúrate de haber ejecutado 'make_dataset.py' primero.")
        return
    except Exception as e:
        print(f"Error al cargar los artefactos de preprocesamiento: {e}")
        return

    # 3. Cargar los nuevos datos crudos
    try:
        new_raw_data = pd.read_excel(new_raw_data_path, sheet_name="SAT HOSP_FEATURE")
        print(f"'{os.path.basename(new_raw_data_path)}' (nuevos datos) cargado correctamente.")
    except FileNotFoundError:
        print(f"Error: Archivo de nuevos datos '{new_raw_data_path}' no encontrado. "
              "Asegúrate de que la ruta sea correcta.")
        return
    except Exception as e:
        print(f"Error al cargar el archivo de nuevos datos Excel: {e}")
        return

    # Guardar los IDs originales si necesitas mapear las predicciones de vuelta
    original_ids = new_raw_data.index if new_raw_data.index.name else pd.Series(range(len(new_raw_data)))

    # 4. Preprocesar los nuevos datos usando las funciones de preprocessing.py y los artefactos cargados
    print("Preprocesando los nuevos datos...")
    df_cleaned = clean_and_impute_data(new_raw_data, edad_median, sat8_mode)
    df_scaled = apply_feature_scaling(df_cleaned, scaler_edad, scaler_dias_hospital)
    
    # Reclasificacion de la variable objetivo
    df_scaled = reclassify_target(df_scaled)   
     
         
    # Para predicción, solo necesitamos X. La función select_features también devuelve y,
    # pero como los datos nuevos no tienen 'sat_general_reclasificada', solo tomaremos X.
    # Necesitamos asegurar que las columnas de X_new coincidan con las de X_train
    # Por la forma en que select_features está definida, debería encargarse de esto.
    X_new, _ = select_features(df_scaled) # El segundo valor (y) será vacío o un error si no existe la columna

    # Para robustez, asegurarse de que X_new tenga las mismas columnas y en el mismo orden que X_train
    # Esto es crítico si el modelo fue entrenado con un orden específico de columnas.
    # Idealmente, las columnas de X_train deberían guardarse como un artefacto.
    # Por ahora, asumiremos que select_features ya las ordena consistentemente.
    # Si hubiera problemas, se podría cargar X_train.csv para obtener el orden de las columnas:
    # X_train_cols = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv')).columns.tolist()
    # X_new = X_new[X_train_cols]
    
    print(f"Nuevos datos preprocesados. Columnas para predicción: {X_new.columns.tolist()}")

    # 5. Realizar predicciones
    print("Realizando predicciones...")
    predictions = model.predict(X_new)
    predictions_proba = model.predict_proba(X_new)[:, 1] # Probabilidad de la clase positiva (1)

    # 6. Guardar las predicciones
    results_df = pd.DataFrame({
        'Original_Index': original_ids, # Para mapear de vuelta si el índice es importante
        'Prediction': predictions,
        'Probability_Class_1': predictions_proba
    })

    output_filename = 'new_data_predictions.csv'
    results_df.to_csv(os.path.join(predictions_dir, output_filename), index=False)
    print(f"Predicciones guardadas correctamente en '{os.path.join(predictions_dir, output_filename)}'.")

    print("Proceso de predicción finalizado.")


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, '..')

    # Ruta de ejemplo para los nuevos datos crudos que se quieren predecir
    # ASUME que tienes un archivo de datos nuevos similar al original en data/raw
    # Por ejemplo, puedes copiar tu archivo original y renombrarlo para probar
    new_raw_data_input_path = os.path.join(project_root, 'data', 'raw', 'DATA DE SATISFACCIÓN HOSPITAL APP_NUEVOS_DATOS.xlsx') # <--- CAMBIA ESTO A LA RUTA DE TU ARCHIVO DE NUEVOS DATOS
    
    models_input_dir = os.path.join(project_root, 'models')
    predictions_output_dir = os.path.join(project_root, 'data', 'predictions')

    run_predict_model(new_raw_data_input_path, models_input_dir, predictions_output_dir)