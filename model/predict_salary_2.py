import joblib
import numpy as np
import pandas as pd

# Cargar modelo optimizado
model = joblib.load("model/xgb_salary_model_2.pkl")
encoder = joblib.load("model/onehot_encoder_2.pkl")
scaler = joblib.load("model/standard_scaler_2.pkl")

# Inflación acumulada (usa el mismo diccionario del entrenamiento)
inflation_map = {
    year: 1.045 ** (year - 2022)
    for year in range(2022, 2031)  # 👈 hasta 2030 incluido
}
# Función para agrupar el título
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

# Función principal
def predict_salary2(job_title: str, experience_level: str, work_year: int) -> dict:
    job_group = agrupar_job_title(job_title)
    exp_year_combo = f"{experience_level}_{work_year}"
    if work_year not in inflation_map:
        raise ValueError(f"Año {work_year} fuera de rango permitido (2022–2030)")
    new_data = pd.DataFrame([{
        'work_year': work_year,
        'experience_level': experience_level,
        'job_group': job_group,
        'employee_residence': 'US',
        'remote_ratio': 0,
        'company_location': 'US',
        'company_size': 'M',
        'exp_year_combo': exp_year_combo
    }])

    # Columnas categóricas usadas en el entrenamiento
    categorical_cols = ['experience_level', 'job_group', 'employee_residence', 'company_location', 'company_size', 'exp_year_combo']
    
    new_encoded = encoder.transform(new_data[categorical_cols])
    new_encoded_df = pd.DataFrame(new_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    new_final = pd.concat([new_data.drop(columns=categorical_cols).reset_index(drop=True), new_encoded_df], axis=1)
    new_scaled = scaler.transform(new_final)

    predicted_log_adjusted = model.predict(new_scaled)
    predicted_adjusted = np.expm1(predicted_log_adjusted)

    # Reaplicar inflación para convertir a salario real del año solicitado
    inflation_factor = inflation_map[work_year]
    predicted_salary = predicted_adjusted[0] * inflation_factor

    return {"predicted_salary": round(float(predicted_salary), 2)}
