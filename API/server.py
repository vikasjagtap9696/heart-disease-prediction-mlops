from contextlib import asynccontextmanager
import os

import pandas as pd
import mlflow
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ------------------------------------------------
# Input Patient Schema
# ------------------------------------------------

class InputPatient(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ------------------------------------------------
# MLflow Configuration
# ------------------------------------------------

model_uri = "models:/heart_disease_prediction/1"
mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]


# ------------------------------------------------
# Application Lifespan
# ------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Set MLflow tracking server
    mlflow.set_tracking_uri(mlflow_uri)

    # Load registered model
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Store model in FastAPI application state
    app.state.model = loaded_model

    print("Model loaded in memory, server is ready to serve requests.")

    yield

    # Cleanup model from memory
    del app.state.model

    print("Server stopped cleanly.")


# ------------------------------------------------
# FastAPI Application
# ------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps-based Heart Disease Prediction API using FastAPI and MLflow",
    version="1.0.0",
    lifespan=lifespan
)


# ------------------------------------------------
# CORS Configuration
# ------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------
# Health Check
# ------------------------------------------------

@app.get("/health", status_code=200)
async def health_check():
    return {"healthy": "true"}


# ------------------------------------------------
# Root Endpoint
# ------------------------------------------------

@app.get("/")
def welcome_message():
    return {
        "message": "FastAPI server is up."
    }


# ------------------------------------------------
# Prediction Endpoint
# ------------------------------------------------

@app.post("/predict")
def predict(data: InputPatient):

    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]

    features = pd.DataFrame(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            data.fbs,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal
        ]],
        columns=columns
    )

    # Generate prediction
    y_pred = app.state.model.predict(features)

    return {
        "prediction": int(y_pred[0])
    }


# ------------------------------------------------
# Run Application
# ------------------------------------------------

if __name__ == "__main__":

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
