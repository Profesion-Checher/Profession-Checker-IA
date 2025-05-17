import pandas as pd
import numpy as np
import json
import joblib
def create_data():
        
    # Cargar modelo, encoder y scaler previamente guardados
    model = joblib.load("model/trained_random_forest_model.pkl")
    encoder = joblib.load("model/onehot_encoder.pkl")
    scaler = joblib.load("model/standard_scaler.pkl")

    # Cargar el DataFrame base
    df = pd.read_csv("model/filtered_data.csv")

    # Extraer combinaciones únicas de títulos de trabajo y niveles de experiencia
    unique_jobs = df['job_title'].unique()
    experience_levels = df['experience_level'].unique()

    # Constantes
    company_location = 'US'
    employee_residence = 'US'
    remote_ratio = 0
    company_size = 'M'
    categorical_cols = ['experience_level', 'job_title', 'employee_residence', 'company_location', 'company_size']

    # Generar entradas JSON
    json_data = []
    pk_counter = 1

    def predict_salary(year, level, job):
        input_data = pd.DataFrame([[
            year, level, job, employee_residence, remote_ratio, company_location, company_size
        ]], columns=['work_year', 'experience_level', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size'])

        encoded = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
        final_input = pd.concat([input_data.drop(columns=categorical_cols).reset_index(drop=True), encoded_df], axis=1)
        final_input = final_input.reindex(columns=scaler.feature_names_in_, fill_value=0)
        final_scaled = scaler.transform(final_input)

        predicted_salary = np.expm1(model.predict(final_scaled)[0])
        return float(predicted_salary)

    # Iterar combinaciones
    for job in unique_jobs:
        for level in experience_levels:
            # Salario actual usando el modelo (año 2025)
            current_salary = predict_salary(2025, level, job)

            # Salarios futuros (2026 a 2028)
            future_salaries = [int(predict_salary(year, level, job)*4000/12) for year in [2026, 2027, 2028, 2029, 2030]]

            json_entry = {
                "model": "professions.profession",
                "pk": pk_counter,
                "fields": {
                    "profession_name": job,
                    "current_salary": round(current_salary*4000/12, 2),
                    "future_salaries": future_salaries,
                    "companies": ["Tuya S.A."],
                    "experience": level
                }
            }

            json_data.append(json_entry)
            pk_counter += 1

    # Guardar archivo
    with open("model/professions.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print("✅ Archivo 'professions.json' generado correctamente.")

create_data()