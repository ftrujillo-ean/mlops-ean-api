# API FastAPI que recibe los datos de un cliente de tarjeta de credito y consulta el endpoint MLOPS_EAN de Databricks
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Credenciales: se inyectan como variables de entorno en Azure (nunca escritas en el código)
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

app = FastAPI(title="API MLOps EAN", description="Predicción de pago de tarjeta de credito")

# Permite que el frontend (index.html) llame a la API desde otro dominio
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# Campos que espera el modelo (deben coincidir con la firma registrada en MLflow)
class CreditCard(BaseModel):
    LIMIT_BAL: int = Field(..., ge=10000, le=1000000)

    # Variables demográficas
    SEX: int = Field(..., ge=0, le=1)
    EDUCATION: int = Field(..., ge=0)
    AGE: int = Field(..., ge=21, le=79)

    # Historial de pagos
    PAY_0: int = Field(..., ge=-2, le=8)
    PAY_2: int = Field(..., ge=-2, le=8)
    PAY_3: int = Field(..., ge=-2, le=8)
    PAY_4: int = Field(..., ge=-2, le=8)
    PAY_5: int = Field(..., ge=-2, le=8)
    PAY_6: int = Field(..., ge=-2, le=8)

    # Montos facturados
    BILL_AMT1: int = Field(..., ge=-165580, le=964511)
    BILL_AMT2: int = Field(..., ge=-69777, le=983931)
    BILL_AMT3: int = Field(..., ge=-157264, le=1664089)
    BILL_AMT4: int = Field(..., ge=-170000, le=891586)
    BILL_AMT5: int = Field(..., ge=-81334, le=927171)
    BILL_AMT6: int = Field(..., ge=-339603, le=961664)

    # Pagos realizados
    PAY_AMT1: int = Field(..., ge=0, le=873552)
    PAY_AMT2: int = Field(..., ge=0, le=1684259)
    PAY_AMT3: int = Field(..., ge=0, le=896040)
    PAY_AMT4: int = Field(..., ge=0, le=621000)
    PAY_AMT5: int = Field(..., ge=0, le=426529)
    PAY_AMT6: int = Field(..., ge=0, le=528666)

    # One-Hot Encoding de estado civil
    MARRIAGE_Single: int = Field(..., ge=0, le=1)
    MARRIAGE_Married: int = Field(..., ge=0, le=1)
    MARRIAGE_Other: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    # Verifica que la API está viva y que tiene configuradas las credenciales
    return {"status": "ok", "databricks_configurado": bool(DATABRICKS_URL and DATABRICKS_TOKEN)}


@app.post("/predecir")
def predecir(est: CreditCard):
    if not (DATABRICKS_URL and DATABRICKS_TOKEN):
        raise HTTPException(status_code=500, detail="Faltan DATABRICKS_URL o DATABRICKS_TOKEN")

    # Databricks exige el formato dataframe_records
    payload = {"dataframe_records": [est.model_dump()]}

    try:
        r = requests.post(
            DATABRICKS_URL,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json=payload,
            timeout=120,  # margen para el arranque en frío del endpoint
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Databricks tardó demasiado (arranque en frío). Intente de nuevo.")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error de Databricks: {r.text}")

    pred = int(r.json()["predictions"][0])  # 1 = aprueba, 0 = reprueba
    return {
        "prediccion": pred,
        "resultado": "Realiza pago ✅" if pred == 1 else "Entra en default ❌",
        "datos_recibidos": est.model_dump(),
    }