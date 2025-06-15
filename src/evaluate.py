# src/evaluate.py

import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_evaluate_model(processed_data_dir, models_dir):
    """
    Carga los datos de prueba procesados y el modelo entrenado,
    realiza predicciones y evalúa el rendimiento del modelo.

    Args:
        processed_data_dir (str): Directorio donde se encuentran los datos procesados.
        models_dir (str): Directorio donde se encuentra el modelo entrenado.
    """
    print("Iniciando la evaluación del modelo...")

    # 1. Cargar los datos de prueba procesados
    try:
        X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv')).squeeze() # .squeeze() para Series
        print("Datos de prueba (X_test, y_test) cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: Archivos de datos de prueba no encontrados en '{processed_data_dir}'. "
              "Asegúrate de haber ejecutado 'make_dataset.py' primero.")
        return
    except Exception as e:
        print(f"Error al cargar los datos de prueba: {e}")
        return

    # 2. Cargar el modelo entrenado
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

    # 3. Realizar predicciones en el conjunto de prueba
    print("Realizando predicciones en el conjunto de prueba...")
    y_pred = model.predict(X_test)
    print("Predicciones generadas.")

    # 4. Evaluación del modelo
    print("\n--- Resultados de la Evaluación del Modelo ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
    print("\nEvaluación del modelo finalizada.")


if __name__ == "__main__":
    # Definir las rutas relativas. Asume que el script se ejecuta desde la raíz del proyecto o desde src/
    current_dir = os.path.dirname(__file__)
    project_root = os.path.join(current_dir, '..') # Asume que 'src' está en la raíz

    processed_data_input_dir = os.path.join(project_root, 'data', 'processed')
    models_input_dir = os.path.join(project_root, 'models')

    run_evaluate_model(processed_data_input_dir, models_input_dir)