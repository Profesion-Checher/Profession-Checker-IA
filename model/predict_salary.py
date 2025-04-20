import joblib
import numpy as np
import pandas as pd

# Cargar los objetos entrenados
model = joblib.load("modelo_rf.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Función para predecir salario
def predict_salary(job_title: str, experience_level: str, work_year: int) -> float:
    input_data = pd.DataFrame([{
        "work_year": work_year,
        "experience_level": experience_level,
        "job_title": job_title,
        "employee_residence": "US",
        "remote_ratio": 0,
        "company_location": "US",
        "company_size": "M"
    }])

    categorical_cols = ['experience_level', 'job_title', 'employee_residence', 'company_location', 'company_size']
    input_encoded = encoder.transform(input_data[categorical_cols])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    input_final = pd.concat([input_data.drop(columns=categorical_cols).reset_index(drop=True), input_encoded_df], axis=1)

    input_final = input_final.reindex(columns=model.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_final)

    predicted_salary = np.expm1(model.predict(input_scaled)[0])
    return round(predicted_salary, 2)
