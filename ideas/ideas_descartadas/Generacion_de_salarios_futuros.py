import pandas as pd
import numpy as np
import itertools
import xgboost as xgb

# Cargar el modelo de XGBoost
modelo = xgb.XGBRegressor()
modelo.load_model('modelo_xgb.json')

# Diccionarios de encoding
job_category_dict = {0: 'AI Research', 1: 'AI Specialist', 2: 'BI & Analytics', 3: 'Business Intelligence', 4: 'Computer Vision & Deep Learning', 5: 'Consulting & Strategy', 6: 'Data Analyst', 7: 'Data Engineer', 8: 'Data Management', 9: 'Data Operations', 10: 'Data Scientist', 11: 'Leadership & Management', 12: 'MLOps & Infrastructure', 13: 'Machine Learning', 14: 'Research & Science', 15: 'Software & Data Architecture'}

experience_level_dict = {0: 'MI', 1: 'SE', 2: 'EN', 3: 'EX'}

# Valores codificados
job_categories = list(job_category_dict.keys())  # 0 a 15
experience_levels = list(experience_level_dict.keys())  # 0 a 3
work_years = [2025, 2026, 2027]  # Próximos 3 años

# Crear todas las combinaciones posibles
combinaciones = list(itertools.product(job_categories, experience_levels, work_years))

# Crear el DataFrame
df_prediccion = pd.DataFrame(combinaciones, columns=['job_category', 'experience_level', 'work_year'])

# Ajustar las nuevas características ingenierizadas
df_prediccion['work_year_x_exp'] = df_prediccion['work_year'] * df_prediccion['experience_level']
df_prediccion['work_year_squared'] = df_prediccion['work_year'] ** 2
df_prediccion['work_year_log'] = np.log1p(df_prediccion['work_year'])

# Ajustar el orden de las columnas según el modelo
orden_columnas = modelo.feature_names_in_
df_prediccion = df_prediccion[orden_columnas]

# Realizar las predicciones
df_prediccion['avg_salary_usd'] = modelo.predict(df_prediccion)

# Decodificar los valores
df_prediccion['job_category'] = df_prediccion['job_category'].map(job_category_dict)
df_prediccion['experience_level'] = df_prediccion['experience_level'].map(experience_level_dict)

# Guardar las predicciones en formato CSV
df_prediccion.to_csv('predicciones_salarios.csv', index=False, encoding='utf-8-sig')
df_prediccion.to_json('predicciones_salarios.json', orient='records', lines=True, force_ascii=False)

print("Predicciones guardadas en 'predicciones_salarios.json'")
print("Predicciones guardadas en 'predicciones_salarios.csv'")
print(df_prediccion.head())
print(f"Total de registros: {len(df_prediccion)}")
