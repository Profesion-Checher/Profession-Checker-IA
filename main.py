from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.predict_salary import predict_salary  # Asegúrate de importar correctamente

app = FastAPI()

# Modelo para el request
class SalaryRequest(BaseModel):
    job_title: str
    experience_level: str
    work_year: int

@app.post("/predict")
async def predict_salary_endpoint(request: SalaryRequest):
    try:
        prediction = predict_salary(
            job_title=request.job_title,
            experience_level=request.experience_level,
            work_year=request.work_year
        )
        return {
            "job_title": request.job_title,
            "experience_level": request.experience_level,
            "work_year": request.work_year,
            "predicted_salary": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")
