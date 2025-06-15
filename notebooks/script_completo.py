import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mord import LogisticAT  # Regresión logística ordinal
import seaborn as sns
import matplotlib.pyplot as plt

# Escalamiento de variables
from sklearn.preprocessing import MinMaxScaler

raw_data = pd.read_excel("./DATA DE SATISFACCIÓN HOSPITAL APP_SESIÓN 03.xlsx", 
                         sheet_name = "SAT HOSP_FEATURE")
						 
# raw_data = raw_data[raw_data["hospital"] == 1]
raw_data.drop(columns = ["hospital","caso","rango_edad"], inplace = True)

raw_data.loc[(raw_data["Tipo_hospitalización"] == 3) & (raw_data["sexo"].isna()), "sexo"] = 2
raw_data.loc[(raw_data["Tipo_hospitalización"] == 1) & (raw_data["sexo"].isna()), "sexo"] = 1
raw_data.loc[(raw_data["Tipo_hospitalización"] == 2) & (raw_data["sexo"].isna()), "sexo"] = 1

edad_median = raw_data["edad"].median()

# Imputando la mediana para la fila en la cual no se indicaba el rango de edad
# la mediana asiganda es de la data antes de la imputación de las edades
raw_data.loc[(raw_data["edad"].isna()), "edad"] = edad_median

# Corrigiendo valores atípicos en la variable "sat8"
# La variable "sat8" representa una escala de satisfacción del 1 al 5
sat8_moda = raw_data["sat8"].mode().iloc[0]
raw_data.loc[raw_data["sat8"] > 5, "sat8"] = sat8_moda


# Completando valores nulos
raw_data.loc[ raw_data["sat8"].isna(), "sat8"] = sat8_moda 

# Supongamos que la columna que deseas escalar se llama 'nombre_columna'
scaler_edad = MinMaxScaler()

# Ajustar y transformar los datos
raw_data['edad_escalada'] = scaler_edad.fit_transform(raw_data[['edad']])

# Supongamos que la columna que deseas escalar se llama 'nombre_columna'
scaler_dias_hospital = MinMaxScaler()

# Ajustar y transformar los datos
raw_data['N_dias_hosp_escalada'] = scaler_dias_hospital.fit_transform(raw_data[['N_días_hosp']])

nivel_satisfaccion = raw_data.copy()
# Definimos un diccionario de mapeo
mapeo_satisfaccion = {
    1: 0,
    2: 0,
    3: 0,
    4: 1, # Satisfecho
    5: 1  # Satisfecho
}

nivel_satisfaccion.loc[:,'sat_general_reclasificada'] = nivel_satisfaccion["sat_general"].map(mapeo_satisfaccion)

# Selección de variables
data_selected_variables = nivel_satisfaccion.copy()

data_selected_variables = data_selected_variables.drop(columns = ['sexo', 'medicion', 
                                                                  'Procedencia', 'Tipo_hospitalización',
                                                                  'N_días_hosp','N_dias_hosp_escalada',
                                                                  'edad' # Se elimina la variable edad con tal de solo
                                                                  # considerar la varialbe edad_escalada la cual es más adecuada para
                                                                  # el modelo
                                                                  ])
																  
# 1. Separar variables predictoras (X) y variable objetivo (y)
X = data_selected_variables.drop(columns=['sat_general', 'sat_general_reclasificada'])
y = data_selected_variables['sat_general_reclasificada']

# 2. Dividir los datos en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar modelo de Regresión Logística Ordinal
model = LogisticAT(alpha=1.0)  # alpha es el parámetro de regularización
model.fit(X_train, y_train)

# 4. Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# 5. Evaluación del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nMatriz de Confusión:\n", confusion_matrix(y_test, y_pred))
