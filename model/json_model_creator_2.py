import pandas as pd
import numpy as np
import json
import joblib
import os

def create_data2():
    # Cargar modelo, encoder y scaler
    model = joblib.load("model/xgb_salary_model_2.pkl")
    encoder = joblib.load("model/onehot_encoder_2.pkl")
    scaler = joblib.load("model/standard_scaler_2.pkl")

    # Cargar el dataset base
    df = pd.read_csv("model/filtered_data.csv")

    # Crear mapa de inflación
    inflation_map = {year: 1.045 ** (year - 2022) for year in range(2022, 2031)}

    # Función para agrupar profesiones
    def agrupar_job_title(title):
        title = title.lower()
        if 'scientist' in title:
            return 'Scientist'
        elif 'engineer' in title:
            return 'Engineer'
        elif 'analyst' in title:
            return 'Analyst'
        elif 'manager' in title:
            return 'Manager'
        else:
            return 'Other'

    # Valores únicos
    unique_jobs = df['job_title'].unique()
    experience_levels = df['experience_level'].unique()

    # Constantes
    company_location = 'US'
    employee_residence = 'US'
    remote_ratio = 0
    company_size = 'M'
    categorical_cols = ['experience_level', 'job_group', 'employee_residence', 'company_location', 'company_size', 'exp_year_combo']

    # JSON resultante
    json_data = []
    pk_counter = 1

    def predict_salary(year, level, job):
        job_group = agrupar_job_title(job)
        exp_year_combo = f"{level}_{year}"

        input_data = pd.DataFrame([{
            'work_year': year,
            'experience_level': level,
            'job_group': job_group,
            'employee_residence': employee_residence,
            'remote_ratio': remote_ratio,
            'company_location': company_location,
            'company_size': company_size,
            'exp_year_combo': exp_year_combo
        }])

        # Codificación y escalado
        encoded = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        final_input = pd.concat([input_data.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1)
        final_input = final_input.reindex(columns=scaler.feature_names_in_, fill_value=0)
        final_scaled = scaler.transform(final_input)

        # Predicción ajustada + corrección por inflación
        predicted_log = model.predict(final_scaled)
        adjusted = np.expm1(predicted_log[0])
        final_salary = adjusted * inflation_map[year]
        return float(final_salary)

    # Iterar combinaciones
    for job in unique_jobs:
        for level in experience_levels:
            current_salary = predict_salary(2025, level, job)
            future_salaries = [int(predict_salary(year, level, job)) for year in [2026, 2027, 2028, 2029, 2030]]

            json_entry = {
                "model": "professions.profession",
                "pk": pk_counter,
                "fields": {
                    "profession_name": job,
                    "current_salary": round(current_salary, 2),
                    "future_salaries": future_salaries,
                    "companies": ["Tuya S.A."],
                    "experience": level
                }
            }
            json_data.append(json_entry)
            pk_counter += 1

    # Guardar archivo JSON
    os.makedirs("model", exist_ok=True)
    with open("model/professions_new.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print("✅ Archivo 'professions_new.json' generado correctamente.")

# Ejecutar
create_data2()
