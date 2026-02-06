from fastapi import FastAPI
from inference.inference import run_inference

app = FastAPI(title="AQI Prediction API")

@app.get("/predict")
def predict():
    return run_inference()
