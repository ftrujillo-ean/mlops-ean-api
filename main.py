# API FastAPI que recibe los datos de un cliente de tarjeta de crédito
# y consulta el endpoint credit-card-default-endpoint de Databricks

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Variables de entorno configuradas en Azure Container Apps
DATABRICKS_URL = os.getenv("DATABRICKS_URL")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Endpoint correcto del modelo de crédito
SERVING_ENDPOINT = "credit-card-default-endpoint"

app = FastAPI(
    title="API MLOps EAN",
    description="Predicción de default de tarjeta de crédito"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


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

    # One-Hot Encoding estado civil
    MARRIAGE_Single: int = Field(..., ge=0, le=1)
    MARRIAGE_Married: int = Field(..., ge=0, le=1)
    MARRIAGE_Other: int = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    endpoint_url = (
        f"{DATABRICKS_URL}/serving-endpoints/"
        f"{SERVING_ENDPOINT}/invocations"
    ) if DATABRICKS_URL else None

    return {
        "status": "ok",
        "databricks_configurado": bool(
            DATABRICKS_URL and DATABRICKS_TOKEN
        ),
        "serving_endpoint": SERVING_ENDPOINT,
        "endpoint_url": endpoint_url
    }


@app.post("/predecir")
def predecir(cliente: CreditCard):

    if not DATABRICKS_URL:
        raise HTTPException(
            status_code=500,
            detail="DATABRICKS_URL no configurada"
        )

    if not DATABRICKS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="DATABRICKS_TOKEN no configurado"
        )

    endpoint_url = (
        f"{DATABRICKS_URL}/serving-endpoints/"
        f"{SERVING_ENDPOINT}/invocations"
    )

    payload = {
        "dataframe_records": [
            cliente.model_dump()
        ]
    }

    try:
        response = requests.post(
            endpoint_url,
            headers={
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Databricks tardó demasiado (cold start)"
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Error conectando con Databricks: {str(e)}"
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Error Databricks [{response.status_code}]: {response.text}"
        )

    resultado = response.json()

    prediccion = int(resultado["predictions"][0])

    return {
        "prediccion": prediccion,
        "resultado": (
            "Realiza pago ✅"
            if prediccion == 0
            else "Entra en default ❌"
        ),
        "modelo": "credit-card-default-endpoint"
    }