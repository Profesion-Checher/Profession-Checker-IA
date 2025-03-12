from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb

app = FastAPI()

# Cargar el modelo de XGBoost
modelo = xgb.XGBRegressor()
modelo.load_model('modelo_xgb.json')

# Diccionarios de encoding
job_category_dict = {0: 'AI Research', 1: 'AI Specialist', 2: 'BI & Analytics', 3: 'Business Intelligence', 4: 'Computer Vision & Deep Learning', 5: 'Consulting & Strategy', 6: 'Data Analyst', 7: 'Data Engineer', 8: 'Data Management', 9: 'Data Operations', 10: 'Data Scientist', 11: 'Leadership & Management', 12: 'MLOps & Infrastructure', 13: 'Machine Learning', 14: 'Research & Science', 15: 'Software & Data Architecture'}

experience_level_dict = {0: 'MI', 1: 'SE', 2: 'EN', 3: 'EX'}

reverse_job_category = {v: k for k, v in job_category_dict.items()}
reverse_experience_level = {v: k for k, v in experience_level_dict.items()}

# Modelo de datos para la solicitud
class SalaryRequest(BaseModel):
    job_category: str
    experience_level: str
    work_year: int

@app.post("/predict")
async def predict_salary(request: SalaryRequest):
    job_category = request.job_category
    experience_level = request.experience_level
    work_year = request.work_year
    
    # Validar las entradas
    if job_category not in reverse_job_category:
        raise HTTPException(status_code=400, detail="job_category no válido")
    if experience_level not in reverse_experience_level:
        raise HTTPException(status_code=400, detail="experience_level no válido")

    # Codificar las entradas
    job_category_encoded = reverse_job_category[job_category]
    experience_level_encoded = reverse_experience_level[experience_level]

    # Crear DataFrame con las características
    data = pd.DataFrame([{
        "job_category": job_category_encoded,
        "experience_level": experience_level_encoded,
        "work_year": work_year
    }])

    # Añadir las características ingenierizadas
    data['work_year_x_exp'] = data['work_year'] * data['experience_level']
    data['work_year_squared'] = data['work_year'] ** 2
    data['work_year_log'] = np.log1p(data['work_year'])

    # Asegurar el orden de las columnas
    data = data[modelo.feature_names_in_]

    # Realizar la predicción
    avg_salary_usd = float(modelo.predict(data)[0])  # Convertir a float de Python

    return {
        "job_category": job_category,
        "experience_level": experience_level,
        "work_year": work_year,
        "avg_salary_usd": avg_salary_usd
    }

print("API de predicción iniciada. Ejecuta con: uvicorn nombre_del_archivo:app --reload")
