# src/train.py

import pandas as pd
import pickle
import os
from mord import LogisticAT # Tu modelo de regresión logística ordinal

def run_train_model(processed_data_dir, models_dir):
    """
    Carga los datos de entrenamiento procesados, entrena el modelo de clasificación
    y guarda el modelo entrenado.

    Args:
        processed_data_dir (str): Directorio donde se encuentran los datos procesados.
        models_dir (str): Directorio donde se guardará el modelo entrenado.
    """
    print("Iniciando el entrenamiento del modelo...")

    # Asegurarse de que el directorio de salida del modelo exista
    os.makedirs(models_dir, exist_ok=True)

    # 1. Cargar los datos de entrenamiento procesados
    try:
        X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'))
        y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv')).squeeze() # .squeeze() para Series
        print("Datos de entrenamiento (X_train, y_train) cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: Archivos de datos procesados no encontrados en '{processed_data_dir}'. "
              "Asegúrate de haber ejecutado 'make_dataset.py' primero.")
        return
    except Exception as e:
        print(f"Error al cargar los datos de entrenamiento: {e}")
        return

    # 2. Entrenar el modelo de Regresión Logística Ordinal
    print("Entrenando el modelo LogisticAT...")
    model = LogisticAT(alpha=1.0) # Usa el mismo parámetro alpha de tu notebook
    model.fit(X_train, y_train)
    print("Modelo entrenado exitosamente.")

    # 3. Guardar el modelo entrenado
    model_path = os.path.join(models_dir, 'best_model.pkl')
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Modelo guardado correctamente en '{model_path}'.")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

    print("Entrenamiento del modelo finalizado.")


if __name__ == "__main__":
    # Definir las rutas relativas. Asume que el script se ejecuta desde la raíz del proyecto o desde src/
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, '..') # Asume que 'src' está en la raíz

    processed_data_input_dir = os.path.join(project_root, 'data', 'processed')
    models_output_dir = os.path.join(project_root, 'models')

    run_train_model(processed_data_input_dir, models_output_dir)