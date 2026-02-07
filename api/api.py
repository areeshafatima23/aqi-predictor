from fastapi import FastAPI
from inference.inference import run_inference

app = FastAPI(title="AQI Prediction API")


@app.get("/predict")
def predict():
    forecast, meta = run_inference()  # best model by RMSE
    model_info = {
        "model_name": meta.get("model_name", "Unknown"),
        "metrics": {
            "rmse": meta.get("metrics", {}).get("rmse", None)
        }
    }
    return {
        "forecast": forecast,
        "model_info": model_info
    }
