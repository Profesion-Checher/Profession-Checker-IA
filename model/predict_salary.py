import joblib
import numpy as np
import pandas as pd

# Cargar los objetos entrenados
model = joblib.load("model/trained_random_forest_model.pkl")
encoder = joblib.load("model/onehot_encoder.pkl")
scaler = joblib.load("model/standard_scaler.pkl")

# Función para predecir salario
def predict_salary(job_title: str, experience_level: str, work_year: int) -> float:
    new_data = np.array([[work_year, experience_level, job_title, 'US', 0, 'US', 'M']])
    new_data_df = pd.DataFrame(new_data, columns=['work_year', 'experience_level', 'job_title', 
                                              'employee_residence', 'remote_ratio', 
                                              'company_location', 'company_size'])

    categorical_cols = ['experience_level', 'job_title', 'employee_residence', 'company_location', 'company_size']
    new_data_encoded = encoder.transform(new_data_df[categorical_cols])
    new_data_encoded_df = pd.DataFrame(new_data_encoded, columns=encoder.get_feature_names_out(categorical_cols))
    new_data_final = pd.concat([new_data_df.drop(columns=categorical_cols).reset_index(drop=True), 
                            new_data_encoded_df], axis=1)
    new_data_scaled = scaler.transform(new_data_final)

    predicted_salary = np.expm1(model.predict(new_data_scaled))
    return {"predicted_salary": float(predicted_salary[0])}